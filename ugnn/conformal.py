import numpy as np
import torch

from typing import Literal
from torch import Tensor
from torch_geometric.data import Data


def get_prediction_sets(
    output: Tensor,
    data: Data,
    calib_mask: Tensor,
    test_mask: Tensor,
    alpha: float = 0.1,
    method: Literal["APS"] = "APS",
) -> np.ndarray[np.bool_]:
    """
    Takes the output of a model and computes the conformal prediction sets.

    This first takes data points from the calibration set and computes a non-conformity score,
    i.e. how "strange" a data-label pair is. Then it computes the quantile of the non-conformity
    scores, which are then used to form the prediction sets for the test set.

    Implemented methods:
    - APS: Adaptive Prediction Sets

    """
    n_calib = calib_mask.sum().item()

    # Compute softmax probabilities
    smx = torch.nn.Softmax(dim=1)
    calib_heuristic = smx(output[calib_mask]).detach().numpy()
    test_heuristic = smx(output[test_mask]).detach().numpy()

    # Compute non-conformity scores
    if method == "APS":
        calib_pi = calib_heuristic.argsort(1)[:, ::-1]
        calib_srt = np.take_along_axis(calib_heuristic, calib_pi, axis=1).cumsum(axis=1)
        calib_scores = np.take_along_axis(calib_srt, calib_pi.argsort(axis=1), axis=1)[
            range(n_calib), data.y[calib_mask]
        ]
    elif method == "RAPS":
        raise NotImplementedError("RAPS method is not implemented yet.")
    else:
        raise ValueError(f"Unknown method: {method}")

    # Get the score quantile
    qhat_quantile = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib

    if qhat_quantile > 1:
        raise ValueError(
            f"Specified quantile is larger than 1. Either increase the number of calibration data points or increase alpha."
        )

    qhat = np.quantile(calib_scores, qhat_quantile, method="higher")

    # Return the prediction sets
    test_pi = test_heuristic.argsort(1)[:, ::-1]
    test_srt = np.take_along_axis(test_heuristic, test_pi, axis=1).cumsum(axis=1)
    pred_sets = np.take_along_axis(test_srt <= qhat, test_pi.argsort(axis=1), axis=1)

    return pred_sets
