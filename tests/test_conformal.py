import pytest
import numpy as np
import torch
from torch_geometric.data import Data

from ugnn.conformal import get_prediction_sets


def test_get_prediction_sets_valid_inputs():
    output = torch.tensor([[2.0, 1.0], [0.5, 1.5], [1.0, 2.0], [1.5, 0.5]])
    data = Data(
        y=torch.tensor([1, 0, 0, 1]),
    )
    calib_mask = torch.tensor([True, True, True, False])
    test_mask = torch.tensor([False, False, False, True])
    pred_sets = get_prediction_sets(
        output,
        data,
        calib_mask,
        test_mask,
        score_function="APS",
        alpha=0.7,
    )

    assert pred_sets.shape == (1, 2)  # Ensure correct shape
    assert pred_sets.dtype == np.bool_  # Ensure binary output


def test_get_prediction_sets_bad_quantile():

    # Same example, but with low n_calibration and low alpha
    # This means we want the m-th quantile of non-conformities but m > 1

    output = torch.tensor([[2.0, 1.0], [0.5, 1.5], [1.0, 2.0], [1.5, 0.5]])
    data = Data(
        y=torch.tensor([1, 0, 0, 1]),
    )
    calib_mask = torch.tensor([True, True, True, False])
    test_mask = torch.tensor([False, False, False, True])

    with pytest.raises(ValueError, match="Specified quantile is larger than 1"):
        get_prediction_sets(
            output,
            data,
            calib_mask,
            test_mask,
            score_function="APS",
            alpha=0.1,
        )


def test_get_prediction_sets_unsupported_method():
    output = torch.tensor([[2.0, 1.0]])
    data = type("Data", (), {"y": torch.tensor([0])})
    calib_mask = torch.tensor([True])
    test_mask = torch.tensor([False])
    with pytest.raises(ValueError, match="Unknown method: INVALID"):
        get_prediction_sets(
            output,
            data,
            calib_mask,
            test_mask,
            score_function="INVALID",
            alpha=0.1,
        )
