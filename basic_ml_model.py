# Basic supervised ML model for learning initial anisotropic mesh generation
# Jason Chen, 17 October, 2024

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import os, re, time
from readgri import readgri
import argparse, psutil, ast

# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 5
TRAIN_FRACTION = 0.8
# NOTE: The following line checking for power only works on Linux systems
DEVICE = ("cuda"
    if torch.cuda.is_available() and psutil.sensors_battery().power_plugged
    else "cpu"
)


class GigaMesher9000(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
        self.spatial_net = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
        self.output_net = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 3)
        )

    def forward(self, x: np.ndarray):
        """
        The feedforward step. x is a 1x6 numpy array, with the first two values being the
        (x,y) position of the centroid of an element, then wall distance, and the next 
        three values representing (Re, Mach, AoA)
        """
        # Since training in batches, take all rows (first arg) and slice by the column
        s_logits = self.spatial_net(x[:, :3])
        p_logits = self.param_net(x[:, 3:])
        # Element-wise multipulcation
        to_feed = (s_logits/torch.norm(s_logits)) * (p_logits/torch.norm(p_logits))
        return self.output_net(to_feed)


class MeshDataset(Dataset):
    def __init__(self, path: str, file_regex: str, args, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.reynolds = 1e6
        self.mach = 0.25
        regex = re.compile(file_regex)
        self.files = [os.path.join(path, f) for f in os.listdir(path) if regex.match(f)]
        self.transform = transform
        # We want to import data, if possible
        if (not args.generate_data) and os.path.exists(args.data_path):
            self.data = pd.read_csv(args.data_path)
            print(f'Training data imported from {args.data_path}')
        else:
            print('Generating training data...')
            start_time = time.perf_counter()
            self.data = self.load_data()
            print(f'Data generated in {time.perf_counter()-start_time:.2f} seconds')
        print(self.data.head())
        if args.export_data:
            print(f'Exporting compiled training data to {args.data_path}')
            self.data.to_csv(args.data_path, index=False)

    def load_data(self) -> pd.DataFrame:
        """
        Generate training data from the meshes. The keys are specified in order below.
        """
        data = []
        data_keys = ['x', 'y', 'wall_dist', 'Re', 'Mach', 'AoA', 'a', 'b', 'c']
        for file in self.files:
            run_num = int(os.path.basename(file).split('.')[0][3:])
            mesh = readgri(file)
            vertices = mesh['V']; elements = mesh['E']
            # Preprocess wall distance information
            dist_f_name = 'walldistance' + str(run_num) + '.txt'
            wall_dists = []
            with open(os.path.join('./walldist', dist_f_name), 'r') as dist_file:
                line = dist_file.readline()
                while line:
                    if line[0] == '%': 
                        pass
                    elif len(line.split()) > 1:
                        avg_d = np.mean([float(dist_file.readline()) for _ in range(3)])
                        wall_dists.append(avg_d)
                    line = dist_file.readline()
            assert(len(wall_dists) == len(elements))

            for i, element in enumerate(elements):
                verts = [vertices[v_ind] for v_ind in element]  # (x,y) for each vertex
                centroid = [np.mean([v[0] for v in verts]), np.mean([v[1] for v in verts])]
                # Combine spatial and parameter information into a feature
                params = [self.reynolds, self.mach, np.deg2rad(run_num-6)]
                feature = centroid + [wall_dists[i]] + params
                # Find the correct A, B, and C terms in our metric field
                delta_x = [verts[i] - verts[i-1] for i in range(len(verts))]
                # All delta x are a unit length L=1 apart, solve for M
                A_mat = [[vec[0]**2, 2*vec[0]*vec[1], vec[1]**2] for vec in delta_x]
                A_mat = np.reshape(np.asarray(A_mat), (3,3))
                metric_abc = np.matmul(np.linalg.inv(A_mat), np.ones((3,1))).flatten()
                label = list(metric_abc)
                # Append to data structure
                data.append(feature + label)
        return pd.DataFrame(data, columns=data_keys)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a tuple in the form of (feature, label) from internal data
        """
        row = tuple(self.data.iloc[idx])
        sample = (row[:6], row[6:])
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor:
    def __call__(self, sample: tuple):
        feature, label = sample
        tensor_feature = torch.tensor(feature, dtype=torch.float64)
        tensor_label = torch.tensor(label, dtype=torch.float64)
        return tensor_feature, tensor_label


class MetricLoss(nn.Module):
    """
    Compute the loss between the predicted metric field and actual metric field.
    We are measuring loss via a symmetric step matrix, S. To do this, we need to
    compute matrix logarithms and powers. Since metric fields are positive definite
    symmetric matrices, we are guaranteed that they are invertible and that we have
    n linearly independent eigenvectors, whose eigenvalues are all real and positive. 
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, prediction: torch.Tensor, actual: torch.Tensor):
        # Not sure if there's a faster way to train a batch besides a for-loop
        total_loss = 0
        for row in range(len(prediction)):
            pred, act = prediction[row], actual[row]
            # Using torch.stack() here to preserve the computation graph for backpropagation
            S_0 = torch.stack([torch.stack([pred[0], pred[1]]), torch.stack([pred[1], pred[2]])])
            M_0 = matrix_operator(S_0, "exp")
            M = torch.stack([torch.stack([act[0], act[1]]), torch.stack([act[1], act[2]])])
            M_0_temp = torch.linalg.inv(matrix_operator(M_0, 1/2))
            S = matrix_operator(M_0_temp @ M @ M_0_temp, "log")
            # Take Frobenius norm of S, ignoring the square root at the end
            total_loss += torch.sum(torch.square(S))
        return total_loss / len(prediction)


def matrix_operator(A: torch.Tensor, operator) -> torch.Tensor:
    """No built-in function for finding matrix exponents and powers."""
    # Using eigh() here since we know A is symmetrical and we only care about real parts.
    e_vals, e_vecs = torch.linalg.eigh(A)
    # Calling .real on Tensors will break the computation graph and disallow backpropagation
    if operator == "exp":
        e_vals = torch.exp(e_vals)
    elif operator == "log":
        e_vals = torch.log(e_vals)
    else:
        assert type(operator) == float or type(operator) == int
        e_vals = torch.pow(e_vals, operator)
    # e_vecs already represents S in A = S*D*S^(-1) since columns are eigenvectors
    return e_vecs @ torch.diag(e_vals) @ torch.linalg.inv(e_vecs)


def train_loop(data_loader: DataLoader, model: nn.Module, loss_fn, optimizer):
    # Set the model to training mode
    model.train()
    for batch_i, (features, labels) in enumerate(data_loader):
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        # Labels is of shape [BATCH_SIZE, 3] since (A,B,C) of metric field
        pred = model(features)  # Passing an entire batch size into the model
        loss = loss_fn(pred, labels)
        # Backpropagation
        loss.backward()
        optimizer.step()
        # Zero the gradients so the next iteration where we calculate losses 
        # and backpropagate does not account for data of this current iteration
        optimizer.zero_grad()
        if batch_i % 50 == 0:
            current, total_data = batch_i * BATCH_SIZE, len(data_loader) * BATCH_SIZE
            print(f"Loss: {loss.item():>7f}  [{current:>5d}/{total_data:>5d}]")


def test_loop(data_loader: DataLoader, model: nn.Module, loss_fn):
    # Set the model to evaluation mode
    model.eval()
    num_batches = len(data_loader)
    test_loss = 0
    # torch.no_grad() ensures that no gradients are computed during test mode
    # Also reduces unnecessary computations for tensors with requires_grad=True
    with torch.no_grad():
        for feature, label in data_loader:
            feature, label = feature.to(DEVICE), label.to(DEVICE)
            pred = model(feature)
            test_loss += loss_fn(pred, label).item()

    test_loss /= num_batches
    print(f"Testing average loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    if DEVICE == 'cuda': print(f"CUDA device: {torch.cuda.get_device_name(DEVICE)}")
    print(f"Device in use: {DEVICE}")
    torch.set_default_dtype(torch.float64)
    torch.set_grad_enabled(True)
    # Ingest command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--export_data', help='Export training data, if generated', action='store_true')
    parser.add_argument('-g', '--generate_data', help='Generate training data instead of importing', action='store_true')
    parser.add_argument('-d', '--data_path', help='Path for training data', default='./data/training_data.csv')
    parser.add_argument('-p', '--plot', help='Plot for debugging', action='store_true', default=False)
    # NOTE: There is a difference between store_false and default=False
    args = parser.parse_args()

    # Format data from .gri files to feed into training and testing
    folder_path = "./meshes/"
    gri_regex = "run.*gri"
    dataset = MeshDataset(folder_path, gri_regex, args, transform=ToTensor())
    train_size = int(TRAIN_FRACTION * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = GigaMesher9000().to(DEVICE)
    loss_fn = MetricLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    for t in range(EPOCHS):
        print(f"----------------Epoch {t+1}---------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)
    print("Done!")
    