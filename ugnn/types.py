from typing import TypedDict, List
import numpy as np


class ExperimentParams(TypedDict):
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
    outputs: List[str]


class DataParams(TypedDict):
    n: int
    T: int
    num_classes: int


class Masks(TypedDict):
    train: np.ndarray
    valid: np.ndarray
    calib: np.ndarray
    test: np.ndarray
