import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
import torch

import torch_geometric
from torch_geometric.data import Dataset

from ugnn.types import AdjacencySeries


class Dynamic_Network(Dataset):
    """
    A pytorch geometric dataset for an n-node discrete-time dynamic network
    with T time points.

    Here a dynamic network consists of:
        As: A collection of T nxn adjacency matrices, shape (T, n, n)
        labels: a list of labels of shape (T, n)

    As we will be using GNNs to embed these networks, we must supply some attributes.
    For this, we use the nxn identity matrix for each time point.
    """

    def __init__(self, As: AdjacencySeries, labels: np.ndarray):
        self.As = As  # series of adjacency matrices
        self.T = As.shape[0]  # number of time points
        self.n = As[0].shape[0]  # number of nodes
        self.classes = labels  # array of labels
        self.sparse = sparse.issparse(As[0])  # check if the matrices are sparse

    def __len__(self):
        return len(self.As)

    def __getitem__(self, idx):
        # This is used to select one of the time points from the dynamic network
        # (the idxth time point)

        # Set attributes to be the identity matrix
        x = form_empty_attributes(self.n, sparse=self.sparse)

        # Get the adjacency matrix for the current time point
        edge_index = torch.tensor(
            np.array([self.As[idx].nonzero()]), dtype=torch.long
        ).reshape(2, -1)
        edge_weight = torch.tensor(np.array(self.As[idx].data), dtype=torch.float)

        # Node labels for this time point
        y = torch.tensor(
            self.classes[self.n * idx : self.n * (idx + 1)], dtype=torch.long
        )

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(
            x=x, edge_index=edge_index, edge_weight=edge_weight, y=y
        )
        data.num_nodes = self.n

        return data


class Block_Diagonal_Network(Dataset):
    """
    A pytorch geometric dataset for the block diagonal version of a Dynamic Network object.

    NOTE: Attributes are left empty (identity) for now.
    Functionality to take attributes will be added soon.
    """

    def __init__(self, dataset: Dataset):
        self.A = block_diagonal_matrix_from_series(dataset.As)
        self.sparse = sparse.issparse(self.A)
        self.T = dataset.T
        self.n = dataset.n
        self.classes = dataset.classes

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = form_empty_attributes(self.n * self.T, sparse=self.sparse)
        edge_index = torch.tensor(
            np.array([self.A.nonzero()]), dtype=torch.long
        ).reshape(2, -1)
        edge_weight = get_edge_weights(self.A)
        y = torch.tensor(self.classes, dtype=torch.long)

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(
            x=x, edge_index=edge_index, edge_weight=edge_weight, y=y
        )
        data.num_nodes = self.n * self.T

        return data


class Unfolded_Network(Dataset):
    """
    A pytorch geometric dataset for the dilated unfolding of a Dynamic Network object.

    NOTE: Attributes are left empty (identity) for now.
    Functionality to take attributes will be added soon.
    """

    def __init__(self, dataset: Dataset):
        self.A = unfolded_matrix_from_series(dataset.As)
        self.sparse = sparse.issparse(self.A)
        self.T = dataset.T
        self.n = dataset.n
        self.classes = dataset.classes

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = form_empty_attributes(self.n * (self.T + 1), sparse=self.sparse)
        edge_index = torch.tensor(
            np.array([self.A.nonzero()]), dtype=torch.long
        ).reshape(2, -1)
        edge_weight = get_edge_weights(self.A)

        # Add n zeros to the start of y for the anchors
        y = torch.tensor(
            np.concatenate((np.zeros(self.n), self.classes)), dtype=torch.long
        )

        # Create a PyTorch Geometric data object
        data = torch_geometric.data.Data(
            x=x, edge_index=edge_index, edge_weight=edge_weight, y=y
        )
        data.num_nodes = self.n * (self.T + 1)

        return data


def get_edge_weights(A):
    """Returns the edge weights for the adjacency matrix A"""
    if sparse.issparse(A):
        return torch.tensor(np.array(A.data), dtype=torch.float)
    else:
        return torch.tensor(A[A != 0], dtype=torch.float)


def form_empty_attributes(m, sparse=False):
    """Forms the appropriately sized identity matrix for the (empty) attributes of the network"""
    if sparse:
        x = torch.sparse.spdiags(torch.ones(m), offsets=torch.tensor([0]), shape=(m, m))
    else:
        x = torch.eye(m, dtype=torch.float)
    return x


def unfolded_matrix_from_series(As: AdjacencySeries):
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


def block_diagonal_matrix_from_series(As: AdjacencySeries):
    """Forms the block diagonal matrix from an adjacency series"""

    # Construct the block diagonal adjacency matrix
    if sparse.issparse(As[0]):
        A = sparse.block_diag(As)
    else:
        A = block_diag(*As)

    return A
