import numpy as np
from unittest.mock import MagicMock

from ugnn.config import EXPERIMENT_PARAMS
from ugnn.experiments import Experiment


def test_experiment_initialisation():

    method = "BD"
    GNN_model = "GCN"
    regime = "transductive"
    data = MagicMock()
    masks = {
        "train": np.array([True, False, False]),
        "valid": np.array([False, True, False]),
        "calib": np.array([False, False, True]),
        "test": np.array([False, False, False]),
    }
    experiment_params = EXPERIMENT_PARAMS
    data_params = {
        "n": 3,
        "T": 2,
        "num_classes": 2,
    }

    experiment = Experiment(
        method=method,
        GNN_model=GNN_model,
        regime=regime,
        data=data,
        masks=masks,
        experiment_params=experiment_params,
        data_params=data_params,
    )

    assert isinstance(experiment, Experiment)
