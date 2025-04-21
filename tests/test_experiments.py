import pytest
import sys
import numpy as np
from unittest.mock import MagicMock, patch

from ugnn.types import AdjacencySeries
from ugnn.experiment_config import SBM_EXPERIMENT_PARAMS
from ugnn.experiments import Experiment, parse_args_load_data


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
    experiment_params = SBM_EXPERIMENT_PARAMS
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


@pytest.mark.parametrize("data", ["school", "sbm", "flight"])
def test_parse_args_load_data(data):

    test_args = [
        "run_conformal_experiment.py",
        "--data",
        data,
        "--debug",
        "--name",
        "test_run",
        "--regime",
        "semi-inductive",
        "--conformal_method",
        "RAPS",
        "--method",
        "BD",
        "--GNN",
        "GCN",
    ]

    with patch.object(sys, "argv", test_args):
        exp_params, data_params = parse_args_load_data()

    # Assertions
    assert exp_params["experiment_name"] == "test_run_debug"
    assert exp_params["regimes"] == ["semi-inductive"]
    assert exp_params["conformal_method"] == "RAPS"
    assert exp_params["methods"] == ["BD"]
    assert exp_params["GNN_models"] == ["GCN"]

    assert data_params["T"] > 0
    assert data_params["n"] > 0
    assert isinstance(data_params["As"], AdjacencySeries)
    assert isinstance(data_params["node_labels"], np.ndarray)
