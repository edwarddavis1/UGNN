import numpy as np
import os
from ugnn.types import ExperimentParams

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MINIMAL_EXPERIMENT_PARAMS: ExperimentParams = {
    "methods": ["UA"],
    "GNN_models": ["GCN"],
    "regimes": ["semi-inductive"],
    "outputs": ["Accuracy", "Avg Size", "Coverage"],
    "data": "school",
    "conformal_method": "APS",
    "num_epochs": 2,
    "learning_rate": 0.01,
    "weight_decay": 5e-4,
    "num_channels_GCN": 8,
    "num_channels_GAT": 8,
    "alpha": 0.1,
    "props": [0.2, 0.1, 0.35, 0.35],
    "num_train_trans": 2,
    "num_permute_trans": 2,
    "num_train_semi_ind": 2,
}


SBM_EXPERIMENT_PARAMS: ExperimentParams = {
    "methods": ["BD", "UA"],
    "GNN_models": ["GCN", "GAT"],
    "regimes": ["semi-inductive", "transductive", "temporal transductive"],
    "outputs": ["Accuracy", "Avg Size", "Coverage"],
    "data": "sbm",
    "conformal_method": "APS",
    "num_epochs": 200,
    "learning_rate": 0.01,
    "weight_decay": 5e-4,
    "num_channels_GCN": 16,
    "num_channels_GAT": 16,
    "alpha": 0.1,
    "props": [0.2, 0.1, 0.35, 0.35],
    "num_train_trans": 10,
    "num_permute_trans": 100,
    "num_train_semi_ind": 50,
}


SCHOOL_EXPERIMENT_PARAMS: ExperimentParams = {
    "methods": ["BD", "UA"],
    "GNN_models": ["GCN", "GAT"],
    "regimes": ["semi-inductive", "transductive", "temporal transductive"],
    "outputs": ["Accuracy", "Avg Size", "Coverage"],
    "data": "school",
    "conformal_method": "APS",
    "num_epochs": 500,
    "learning_rate": 0.01,
    "weight_decay": 5e-4,
    "num_channels_GCN": 16,
    "num_channels_GAT": 16,
    "alpha": 0.1,
    "props": [0.2, 0.1, 0.35, 0.35],
    "num_train_trans": 10,
    "num_permute_trans": 100,
    "num_train_semi_ind": 50,
}

FLIGHT_EXPERIMENT_PARAMS: ExperimentParams = {
    "methods": ["BD", "UA"],
    "GNN_models": ["GCN", "GAT"],
    "regimes": ["semi-inductive", "transductive", "temporal transductive"],
    "outputs": ["Accuracy", "Avg Size", "Coverage"],
    "data": "flight",
    "conformal_method": "APS",
    "num_epochs": 30,
    "learning_rate": 0.01,
    "weight_decay": 5e-4,
    "num_channels_GCN": 32,
    "num_channels_GAT": 32,
    "alpha": 0.1,
    "props": [0.2, 0.1, 0.35, 0.35],
    "num_train_trans": 10,
    "num_permute_trans": 100,
    "num_train_semi_ind": 50,
}


def validate_params(params):
    """
    Validate the experiment parameters.

    Args:
        params (dict): Dictionary of experiment parameters.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if params["num_epochs"] <= 0:
        raise ValueError("num_epochs must be a positive integer.")
    if not (0 < params["learning_rate"] < 1):
        raise ValueError("learning_rate must be between 0 and 1.")
    if params["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative.")
    if params["num_channels_GCN"] <= 0 or params["num_channels_GAT"] <= 0:
        raise ValueError(
            "num_channels_GCN and num_channels_GAT must be positive integers."
        )
    if np.sum(params["props"]) != 1:
        raise ValueError("props must sum to 1.")

    if not all(method in ["BD", "UA"] for method in params["methods"]):
        raise ValueError("methods must be either 'BD' or 'UA'.")


validate_params(MINIMAL_EXPERIMENT_PARAMS)
validate_params(SBM_EXPERIMENT_PARAMS)
validate_params(SCHOOL_EXPERIMENT_PARAMS)
validate_params(FLIGHT_EXPERIMENT_PARAMS)
