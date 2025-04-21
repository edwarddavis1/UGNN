from typing import TypedDict, List, Union
from scipy.sparse import csr_matrix
import numpy as np


class ExperimentParams(TypedDict):
    experiment_name: str
    num_epochs: int
    learning_rate: float
    weight_decay: float
    num_channels_GCN: int
    num_channels_GAT: int
    alpha: float
    props: List[float]
    num_train_trans: int
    num_permute_trans: int
    num_train_semi_ind: int
    methods: List[str]
    GNN_models: List[str]
    regimes: List[str]


AdjacencySeries = Union[np.ndarray, List[csr_matrix]]

class DataParams(TypedDict):
    As: AdjacencySeries
    node_labels: np.ndarray
    n: int
    T: int
    num_classes: int


class Masks(TypedDict):
    train: np.ndarray
    valid: np.ndarray
    calib: np.ndarray
    test: np.ndarray
