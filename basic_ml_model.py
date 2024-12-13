# Basic supervised ML model for learning initial anisotropic mesh generation
# Jason Chen, 17 October, 2024

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os, re, time
from readgri import readgri
import argparse, psutil

# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 20
MOMENTUM = 0.9  # Allows optimizer to overcome local minima
TRAIN_FRACTION = 0.8
# NOTE: The following line checking for power only works on Linux systems
DEVICE = ("cuda"
    if torch.cuda.is_available() and psutil.sensors_battery().power_plugged
    else "cpu"
)
DATA_KEYS = ['x', 'y', 'wall_dist', 'Re', 'Mach', 'AoA', 'a', 'b', 'c']


class GigaMesher9000(nn.Module):
    def __init__(self, act_fn: nn.Module = nn.ReLU()):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(3, 32),
            act_fn,
            nn.Linear(32, 32),
            act_fn,
            nn.Linear(32, 10)
        )
        self.spatial_net = nn.Sequential(
            nn.Linear(3, 64),
            act_fn,
            nn.Linear(64, 128),
            act_fn,
            nn.Linear(128, 64),
            act_fn,
            nn.Linear(64, 10)
        )
        self.output_net = nn.Sequential(
            nn.Linear(10, 128),
            act_fn,
            nn.Linear(128, 512),
            act_fn,
            nn.Linear(512, 512),
            act_fn,
            nn.Linear(512, 256),
            act_fn,
            nn.Linear(256, 64),
            act_fn,
            nn.Linear(64, 3)
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
        """
        A torch-optimized feedforward call. At the end, since the actual
        matrices are 2x2, we want to create a 3D tensor of BATCH_SIZE of
        2x2 matrices. So, we first stack the first row of the 2x2, which
        is [a, b], into a Tensor of shape (BATCH_SIZE, 2). The same is
        done with [c, d], which also creates a (BATCH_SIZE, 2) Tensor.
        Then, we want to call torch.stack() with dim = -2, signifying we
        want to insert/combine columns at the -2 index position, creating
        an S_0 tensor of size (BATCH_SIZE, 2, 2).
        """
        # Reshape predictions and actuals to create batch matrices
        S_0 = torch.stack([
            torch.stack([prediction[:, 0], prediction[:, 1]], dim=-1),
            torch.stack([prediction[:, 1], prediction[:, 2]], dim=-1)
        ], dim=-2)  # Shape: (batch_size, 2, 2)
        M = torch.stack([
            torch.stack([actual[:, 0], actual[:, 1]], dim=-1),
            torch.stack([actual[:, 1], actual[:, 2]], dim=-1)
        ], dim=-2)  # Shape: (batch_size, 2, 2)

        # Raw matrix output is the matrix log of the metric, so exponentiate
        M_0 = matrix_operator(S_0, "exp")
        M_0_temp = torch.linalg.inv(matrix_operator(M_0, 1/2))
        # Compute S matrix, which measures distances between metrics
        transformed_M = M_0_temp @ M @ M_0_temp
        S = matrix_operator(transformed_M, "log")
        # Compute Frobenius norm of S for each batch element
        frobenius_norm_squared = torch.sum(S**2, dim=(-2, -1))
        # First sum along -2 (columns), then by rows (-1)
        return torch.mean(frobenius_norm_squared)


def plot_loss(mesh_file: str, data_set: MeshDataset, model: nn.Module, loss_fn):
    """
    Plot the ellipses corresponding to the prediction of the model, and the training
    data label (truth) for a given loss. This is to verify the data and loss function
    are performing as expected. Recall that the output of the network is S, for which
    we need to take the matrix exponent of to get the actual metric field prediction 
    containing the elements (A, B, C).
    """
    # Plot the bare mesh
    mesh = readgri(mesh_file)
    V = mesh['V']; E = mesh['E']; BE = mesh['BE']
    fig, ax = plt.subplots()
    plt.triplot(V[:,0], V[:,1], E, 'k-')
    for i in range(BE.shape[0]):
        plt.plot(V[BE[i,0:2],0], V[BE[i,0:2],1], '-', linewidth=1, color='black')

    # Generate loss from this entire GRI file
    base_name = os.path.basename(mesh_file).split('.')[0]
    start, end = data_set.dataset.data_split_indices[base_name]
    # Find the data for this particular GRI file within our combined dataset
    data_rows = data_set.dataset.data.iloc[start: end]
    features = torch.from_numpy(data_rows.iloc[:, :6].to_numpy()).to(DEVICE)
    labels = torch.from_numpy(data_rows.iloc[:, 6:].to_numpy()).to(DEVICE)
    predictions, loss = None, None
    model.eval()
    with torch.no_grad():
        # Run the model in evaluation mode on all features, and record the loss
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


def plot_individual_loss(mesh_file: str, data_set: MeshDataset, model: nn.Module, loss_fn):
    """
    Plot losses of individual elements, one at a time.
    """
    WINDOW = (-0.2, 0.2, -0.3, -0.1)
    mesh = readgri(mesh_file)
    V = mesh['V']; E = mesh['E']; BE = mesh['BE']
    fig, ax = plt.subplots()
    plt.triplot(V[:,0], V[:,1], E, 'k-')
    for i in range(BE.shape[0]):
        plt.plot(V[BE[i,0:2],0], V[BE[i,0:2],1], '-', linewidth=1, color='black')

    base_name = os.path.basename(mesh_file).split('.')[0]
    start, end = data_set.dataset.data_split_indices[base_name]
    data_rows = data_set.dataset.data.iloc[start: end]

    plt.style.use('fast')
    plt.ion()
    plt.axis(WINDOW)
    fig = plt.gcf()
    fig.set_size_inches(16, 9, forward=True)
    for i in range(len(E)):
        row = data_rows.iloc[i]
        saved_centroid = [row['x'], row['y']]
        x_min, x_max, y_min, y_max = WINDOW
        if not((x_min < row['x'] < x_max) and (y_min < row['y'] < y_max)):
            continue
        # Get and plot the correct ellipse data
        soln_M_mat = torch.tensor([[row['a'], row['b']], [row['b'], row['c']]])
        # Make the prediction
        model.eval()
        curr_pred = None
        curr_label = torch.from_numpy(row[6:].to_numpy().reshape(1,3)).to(DEVICE)
        with torch.no_grad():
            curr_element_feat = torch.from_numpy(row[:6].to_numpy().reshape(1,6)).to(DEVICE)
            curr_pred = model(curr_element_feat)
        S_0 = torch.tensor([[curr_pred[0,0], curr_pred[0,1]], [curr_pred[0,1], curr_pred[0,2]]])
        pred_M_mat = matrix_operator(S_0, "exp")
        loss = loss_fn(curr_pred, curr_label)
        # Also, plot a general average ellipse to sanity check
        control_mat = torch.tensor([[223.0, 4.0], [4.0, 450.0]])
        control_log = matrix_operator(control_mat, 'log').flatten()
        control_raw_output = torch.tensor([[control_log[0], control_log[1], control_log[-1]]])
        control_loss = loss_fn(control_raw_output.to(DEVICE), curr_label)

        plot_loss_helper(saved_centroid, soln_M_mat, ax, 'blue')
        plot_loss_helper(saved_centroid, control_mat, ax, 'green')
        plot_loss_helper(saved_centroid, pred_M_mat, ax, 'red')
        plt.show()
        plt.pause(0.1)
        print('------------------------------------')
        print(f'Centroid at ({row['x']:.3f}, {row['y']:.3f}) and wall distance of {row['wall_dist']:.3f}')
        print(f'True metric: {soln_M_mat}')
        print(f'Predicted metric: {pred_M_mat}')
        print(f'Loss for this element: {loss:.2f}')
        print(f'Reference (control) loss: {control_loss:.2f}')
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
    return e_vecs @ torch.diag_embed(e_vals) @ torch.linalg.inv(e_vecs)


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
    return test_loss


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
    parser.add_argument('-i', '--interactive', help='Interactively debug', action='store_true', default=False)
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = GigaMesher9000().to(DEVICE)
    loss_fn = MetricLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',        # Minimize validation loss
        factor=0.2,        # Multiply LR by 0.2 when triggered
        patience=2,        # Wait 2 epochs of no improvement before reducing LR
        threshold=5.0,     # Minimum change to qualify as an improvement
        cooldown=0,        # Wait 0 epochs after reducing LR before monitoring again
        min_lr=1e-6        # Do not reduce below this learning rate
    )
    
    for epoch in range(EPOCHS):
        print(f"----------------Epoch {epoch+1} of {EPOCHS}----------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        val_loss = test_loop(test_loader, model, loss_fn)
        scheduler.step(val_loss)
        # Print diagnostic information
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: val_loss={val_loss:.3f}, learning_rate={current_lr:.6f}")
        if args.interactive:
            plot_individual_loss(os.path.join(folder_path, debug_mesh), train_dataset, model, loss_fn)
        elif args.plot:
            plot_loss(os.path.join(folder_path, debug_mesh), train_dataset, model, loss_fn)
    print('Program complete!')
