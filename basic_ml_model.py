# Basic supervised ML model for learning initial anisotropic mesh generation
# Jason Chen, 17 October, 2024

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os, re, time
from readgri import readgri
import argparse, psutil

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
DATA_KEYS = ['x', 'y', 'wall_dist', 'Re', 'Mach', 'AoA', 'a', 'b', 'c']


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

    def forward(self, x: torch.Tensor):
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
        super(MeshDataset, self).__init__()
        self.reynolds = float(np.log(1e6))
        self.mach = 0.25
        regex = re.compile(file_regex)
        self.files = [os.path.join(path, f) for f in os.listdir(path) if regex.match(f)]
        self.files = sorted(self.files, key=lambda x: int(os.path.basename(x).split('.')[0][3:]))
        self.transform = transform
        self.data_split_indices = dict()

        # We want to import data, if possible
        base_name = os.path.basename(args.data_path).split('.')[0]
        info_path = args.data_path.replace(base_name, base_name + '_info')
        if (not args.generate_data) and os.path.exists(args.data_path) and os.path.exists(info_path):
            start_time = time.perf_counter()
            self.data = pd.read_csv(args.data_path)
            temp_df = pd.read_csv(info_path, index_col=0)
            self.data_split_indices = {i: tuple(row) for i, row in temp_df.iterrows()}
            time_elapsed = time.perf_counter()-start_time
            print(f'Training data imported from {args.data_path} in {time_elapsed:.2f} seconds')
        else:
            print('Generating training data...')
            start_time = time.perf_counter()
            self.data = self.create_data()
            print(f'Data generated in {time.perf_counter()-start_time:.2f} seconds')
        print(self.data.head())
        if args.export_data:
            print(f'Exporting compiled training data to {args.data_path}')
            self.data.to_csv(args.data_path, index=False)
            pd.DataFrame.from_dict(self.data_split_indices, orient='index').to_csv(info_path)

    def create_data(self) -> pd.DataFrame:
        """
        Generate training data from the meshes. Also creates an index for the data structure.
        """
        data = []
        for file in self.files:
            run_str = os.path.basename(file).split('.')[0]
            run_num = int(run_str[3:])
            mesh = readgri(file)
            vertices = mesh['V']; elements = mesh['E']
            # Preprocess wall distance information
            dist_f_name = 'walldistance' + str(run_num) + '.txt'
            wall_dists = []
            with open(os.path.join('./fine_walldist', dist_f_name), 'r') as dist_file:
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

            # Add the split points if we would like to verify training data
            start_i = 0
            if f'run{run_num-1}' in self.data_split_indices.keys():
                start_i =  self.data_split_indices[f'run{run_num-1}'][-1]
            self.data_split_indices.update({run_str: (start_i, start_i + len(elements))})
        return pd.DataFrame(data, columns=DATA_KEYS)

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


def plot_loss(mesh_file: str, data_set: MeshDataset, model: nn.Module, loss_fn):
    """
    Plot the ellipses corresponding to the prediction of the model, and the training
    data label (truth) for a given loss. This is to verify the data and loss function
    are performing as expected. 
    """
    # Recall that the output of the network is S, for which we need to take the matrix
    # exponent of to get the actual metric field prediction containing (A, B, C)
    plt.style.use('fast')
    mesh = readgri(mesh_file)
    # Plot the bare mesh
    V = mesh['V']; E = mesh['E']; BE = mesh['BE']
    fig, ax = plt.subplots()
    plt.triplot(V[:,0], V[:,1], E, 'k-')
    for i in range(BE.shape[0]):
        plt.plot(V[BE[i,0:2],0], V[BE[i,0:2],1], '-', linewidth=1, color='black')

    # Generate loss from this entire GRI file
    base_name = os.path.basename(mesh_file).split('.')[0]
    start, end = data_set.dataset.data_split_indices[base_name]
    data_rows = data_set.dataset.data.iloc[start: end]
    features = torch.from_numpy(data_rows.iloc[:, :6].to_numpy()).to(DEVICE)
    labels = torch.from_numpy(data_rows.iloc[:, 6:].to_numpy()).to(DEVICE)
    predictions, loss = None, None
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        loss = loss_fn(predictions, labels)

    # Plot the ellipses
    for i, element in enumerate(E):
        if i % 50 != 0: continue
        verts = [V[v_ind] for v_ind in element]
        raw_centroid = [np.mean([v[0] for v in verts]), np.mean([v[1] for v in verts])]
        row = data_rows.iloc[i]
        saved_centroid = [row['x'], row['y']]
        # Check that the centroid from training data is the same as seen in the file
        assert(np.all(np.isclose(raw_centroid, saved_centroid)))

        # Get and plot the correct ellipse data
        soln_M_mat = torch.tensor([[row['a'], row['b']], [row['b'], row['c']]])
        plot_loss_helper(saved_centroid, soln_M_mat, ax, 'blue')

        # Get and plot the prediction ellipse
        pred_row = predictions[i]
        S_0 = torch.tensor([[pred_row[0], pred_row[1]], [pred_row[1], pred_row[2]]])
        pred_M_mat = matrix_operator(S_0, "exp")
        plot_loss_helper(saved_centroid, pred_M_mat, ax, 'red')

    plt.axis('equal')
    plt.title(f'Prediction error = {loss:.2f} for {base_name}')
    plt.tick_params(axis='both', labelsize=12)
    fig.tight_layout()
    # Create dummy legend handles for the ellipses
    red_patch = Ellipse((0, 0), 1, 1, color='red', label='Prediction')
    blue_patch = Ellipse((0, 0), 1, 1, color='blue', label='Actual')
    # Add these handles to the legend
    handles = [red_patch, blue_patch]
    labels = [handle.get_label() for handle in handles]
    ax.legend(handles=handles, labels=labels, loc='upper right')
    plt.show()
    plt.close(fig)


def plot_loss_helper(centroid, M_mat: torch.Tensor, ax, color: str):
    eig_vals, eig_vecs = np.linalg.eig(M_mat.cpu())
    eig_vecs = np.asarray(eig_vecs)
    major_ax_i = np.argmax(eig_vals)
    minor_ax_i = int(not major_ax_i)
    axes_scales = np.zeros(len(eig_vals))
    axes_scales[major_ax_i] = eig_vals[major_ax_i]**(-1/2)
    axes_scales[minor_ax_i] = eig_vals[minor_ax_i]**(-1/2)
    angle_major = np.arctan2(eig_vecs[major_ax_i][1], eig_vecs[major_ax_i][0])
    plt.scatter(centroid[0], centroid[1], color='red', alpha=0.5, s=2)
    ellipse = Ellipse(centroid, axes_scales[major_ax_i], axes_scales[minor_ax_i],
                        angle=-angle_major*180/np.pi, facecolor=color, alpha=0.5)
    ax.add_patch(ellipse)


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
    folder_path = "./fine_meshes/"
    gri_regex = "run.*gri"
    debug_mesh = "run1.gri"
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
        if args.plot:
            plot_loss(os.path.join(folder_path, debug_mesh), train_dataset, model, loss_fn)
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)
    print("Done!")
