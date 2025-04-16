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
    Computes conformal prediction sets from the model's output.

    This function uses data points from the calibration set to compute a non-conformity score,
    which measures how "strange" a data-label pair is. It then calculates the quantile of the
    non-conformity scores to form prediction sets for the test set.

    Args:
        output (Tensor): The model's output logits or probabilities.
        data (Data): The data object containing labels (`data.y`) and other graph-related information.
        calib_mask (Tensor): A boolean mask indicating the calibration set.
        test_mask (Tensor): A boolean mask indicating the test set.
        alpha (float, optional): Error rate. Defaults to 0.1.
        method (Literal["APS"], optional): The method to compute prediction sets. Currently, only
            "APS" (Adaptive Prediction Sets) is implemented. Defaults to "APS".

    Returns:
        np.ndarray[np.bool_]: A binary NumPy array where each row corresponds to a test sample,
        and each column indicates whether a class is included in the prediction set.

    Raises:
        NotImplementedError: If the specified method is "RAPS", which is not implemented.
        ValueError: If an unknown method is specified.
        ValueError: If the computed quantile is greater than 1, which indicates insufficient
            calibration data or an invalid alpha value.

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
