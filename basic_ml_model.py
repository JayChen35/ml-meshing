# Basic supervised ML model for learning initial anisotropic mesh generation
# Jason Chen, 17 October, 2024

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.linalg import logm
import pandas as pd
import os, re
from readgri import readgri

# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 5


class GigaMesher9000(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
        self.spatial_net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
        self.output_net = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )


    def forward(self, x: np.ndarray):
        """
        The feedforward step. x is a 1x5 numpy array, with the first two values being the
        (x,y) position of a point, and the next three values representing (Re, Mach, AoA)
        """
        # Since training in batches, take all rows (first arg) and slice by the column
        s_logits = self.spatial_net(x[:, :2])
        p_logits = self.param_net(x[:, 2:])
        # Element-wise multipulcation
        to_feed = (s_logits/torch.norm(s_logits)) * (p_logits/torch.norm(p_logits))
        return self.output_net(to_feed)


class MeshDataset(Dataset):
    def __init__(self, path: str, file_regex: str, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.reynolds = 1e6
        self.mach = 0.25
        regex = re.compile(file_regex)
        self.files = [os.path.join(path, f) for f in os.listdir(path) if regex.match(f)]
        self.transform = transform
        self.data = self.load_data()

    def load_data(self) -> pd.DataFrame:
        data = []
        for file in self.files:
            run_num = int(os.path.basename(file).split('.')[0][3:])
            mesh = readgri(file)
            vertices = mesh['V']; elements = mesh['E']
            for element in elements:
                verts = [vertices[v_ind] for v_ind in element]  # (x,y) for each vertex
                centroid = [np.mean([v[0] for v in verts]), np.mean([v[1] for v in verts])]
                # Combine spatial and parameter information into a feature Tensor 
                feature = centroid + [self.reynolds, self.mach, np.deg2rad(run_num-6)]
                
                # Find the correct A, B, and C terms in our metric field
                delta_x = [verts[i] - verts[i-1] for i in range(len(verts))]
                # All delta x are a unit length L=1 apart, solve for M
                A_mat = [[vec[0]**2, 2*vec[0]*vec[1], vec[1]**2] for vec in delta_x]
                A_mat = np.reshape(np.asarray(A_mat), (3,3))
                metric_abc = np.matmul(np.linalg.inv(A_mat), np.ones((3,1))).flatten()
                # Take the matrix logarithm to reduce the wide range of possible values
                metric_field = [[metric_abc[0], metric_abc[1]], [metric_abc[1], metric_abc[2]]]
                label = matrix_log(torch.tensor(metric_field)).flatten().numpy()
                # Define a single sample of our (feature, label) pair
                data.append((np.asarray(feature), label))
        return pd.DataFrame(data, columns=['feature', 'label'])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a tuple in the form of (feature, label) from internal data
        """
        sample = tuple(self.data.iloc[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor:
    def __call__(self, sample: tuple):
        feature, label = sample
        tensor_feature = torch.tensor(feature, dtype=torch.float32)
        tensor_label = torch.tensor(label, dtype=torch.float32)
        return tensor_feature, tensor_label


def matrix_loss(prediction: torch.Tensor, actual: torch.Tensor):
    """
    Compute the loss between the predicted metric field and actual metric field.
    We are measuring loss via a symmetric step matrix, S. To do this, we need to
    compute matrix logarithms and powers. Since metric fields are positive definite
    symmetric matrices, we are guaranteed that they are invertible and that we have
    n linearly independent eigenvectors, whose eigenvalues are all real and positive. 
    """
    # Not sure if there's a faster way to train a batch besides a for-loop
    total_loss = 0
    for row in range(len(prediction)):
        M_0 = torch.unflatten(prediction[row], 0, (2,2))
        M = torch.unflatten(actual[row], 0, (2,2))
        M_0_temp = torch.linalg.inv(matrix_to_power(M_0, 1/2))
        S = matrix_log(M_0_temp @ M @ M_0_temp)
        total_loss += torch.diagonal(S)
    return total_loss / len(prediction)


def matrix_to_power(A: torch.Tensor, n: float) -> torch.Tensor:
    """No built-in function for finding a matrix to the nth power."""
    e_vals, e_vecs = torch.linalg.eig(A)
    # Extract only real parts of eigenvalues
    e_vals = torch.tensor([x.real.item()**n for x in e_vals])
    e_vecs = e_vecs.real
    # e_vecs already represents S in A = S*D*S^(-1) since columns are eigenvectors
    return e_vecs @ torch.diag(e_vals) @ torch.linalg.inv(e_vecs)


def matrix_log(A: torch.Tensor) -> torch.Tensor:
    """No built-in function for finding the logarithm of a matrix."""
    e_vals, e_vecs = torch.linalg.eig(A)
    e_vals = torch.tensor([np.log(x.real.item()) for x in e_vals])
    e_vecs = e_vecs.real
    return e_vecs @ torch.diag(e_vals) @ torch.linalg.inv(e_vecs)


def train_loop(data_loader: DataLoader, model: nn.Module, loss_fn, optimizer, device):
    # Set the model to training mode
    model.train()
    for batch_i, sample_batched in enumerate(data_loader):
        features, labels = sample_batched
        # Labels is of shape [BATCH_SIZE, 3] since (A,B,C) of metric field
        pred = model(features.to(device))  # Passing an entire batch size into the model
        loss = loss_fn(pred, labels.to(device))
        # Backpropagation
        loss.backward()
        optimizer.step()
        # Zero the gradients so the next iteration where we calculate losses 
        # and backpropagate does not account for data of this current iteration
        optimizer.zero_grad()
        if batch_i % 50 == 0:
            current, total_data = batch_i * BATCH_SIZE, len(data_loader) * BATCH_SIZE
            print(f"Loss: {loss.item():>7f}  [{current:>5d}/{total_data:>5d}]")


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # Format data from .gri files to feed into training and testing
    folder_path = "./meshes/"
    gri_regex = "run.*gri"
    data_set = MeshDataset(folder_path, gri_regex, transform=ToTensor())
    data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = GigaMesher9000().to(device)
    loss_fn = matrix_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(data_loader, model, loss_fn, optimizer, device)
        # test(test_dataloader, model, loss_fn)
    print("Done!")
    