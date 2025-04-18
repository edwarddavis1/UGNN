import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
import torch

import torch_geometric
from torch_geometric.data import Dataset


class Dynamic_Network(Dataset):
    """
    A pytorch geometric dataset for an n-node discrete-time dynamic network
    with T time points.

    Here a dynamic network consists of:
        As: a list of adjacency matrices of shape (T, n, n)
        labels: a list of labels of shape (T, n)

    As we will be using GNNs to embed these networks, we must supply some attributes.
    For this, we use the nxn identity matrix for each time point.
    """

    def __init__(self, As, labels):
        self.As = As  # list of adjacency matrices
        self.T = As[0]  # number of time points
        self.n = As[0].shape[0]  # number of nodes
        self.classes = labels  # list of labels

    def __len__(self):
        return len(self.As)

    def __getitem__(self, idx):
        # This is used to select one of the time points from the dynamic network
        # (the idxth time point)

        # Set attributes to be the identity matrix
        x = torch.tensor(np.eye(self.n), dtype=torch.float)

        # Get the adjacency matrix for the current time point
        edge_index = torch.tensor(
            np.array([self.As[idx].nonzero()]), dtype=torch.long
        ).reshape(2, -1)

        # Node labels for this time point
        y = torch.tensor(self.classes, dtype=torch.long)

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = self.n

        return data


class Block_Diagonal_Network(Dataset):
    """
    A pytorch geometric dataset for the block diagonal version of a Dynamic Network object.
    """

    def __init__(self, dataset):
        self.A = block_diag(*dataset.As)
        self.T = dataset.T
        self.n = dataset.n
        self.classes = dataset.classes

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = torch.tensor(np.eye(self.n * self.T), dtype=torch.float)
        edge_index = torch.tensor(
            np.array([self.A.nonzero()]), dtype=torch.long
        ).reshape(2, -1)
        y = torch.tensor(self.classes, dtype=torch.long)

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = self.n * self.T

        return data


class Unfolded_Network(Dataset):
    """
    A pytorch geometric dataset for the dilated unfolding of a Dynamic Network object.
    """

    def __init__(self, dataset):
        self.A = general_unfolded_matrix(dataset.As)
        self.T = dataset.T
        self.n = dataset.n
        self.classes = dataset.classes

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = torch.tensor(np.eye(self.n * (self.T + 1)), dtype=torch.float)
        edge_index = torch.tensor(
            np.array([self.A.nonzero()]), dtype=torch.long
        ).reshape(2, -1)
        y = torch.tensor(
            np.concatenate((np.zeros(self.n), self.classes)), dtype=torch.long
        )

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = self.n * (self.T + 1)

        return data


def general_unfolded_matrix(As):
    """Forms the general unfolded matrix from an adjacency series"""
    T = len(As)
    n = As[0].shape[0]

    # Construct the rectangular unfolded adjacency
    if sparse.issparse(As[0]):
        A = As[0]
        for t in range(1, T):
            A = sparse.hstack((A, As[t]))

        # Construct the dilated unfolded adjacency matrix
        DA = sparse.bmat([[None, A], [A.T, None]])
        DA = sparse.csr_matrix(DA)
    else:
        A = As[0]
        for t in range(1, T):
            A = np.block([A, As[t]])

        DA = np.zeros((n + n * T, n + n * T))
        DA[0:n, n:] = A
        DA[n:, 0:n] = A.T

    return DA
