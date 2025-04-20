import numpy as np
import torch

from typing import Literal
from torch import Tensor
from torch_geometric.data import Data


def get_prediction_sets(
    output: Tensor,
    data: Data,
    calib_mask: Tensor | np.ndarray,
    test_mask: Tensor | np.ndarray,
    score_function: Literal["APS", "RAPS", "SAPS", "THR"] = "APS",
    alpha=0.1,
    kreg=1,
):
    """
    Computes prediction sets for a given model's output using conformal prediction.

    This function uses data points from the calibration set to compute a non-conformity score,
    which measures how "strange" a data-label pair is. It then calculates the quantile of the
    non-conformity scores to form prediction sets for the test set.

    Args:
        output : Tensor
            The model's output, typically softmax probabilities or logits, for all nodes or samples.
        data : Data
            The dataset object containing features, labels, and other graph-related information.
        calib_mask : Tensor
            A boolean mask indicating which samples belong to the calibration set.
        test_mask : Tensor
            A boolean mask indicating which samples belong to the test set.
        score_function : Literal["APS", "RAPS", "SAPS", "THR"], optional
            The scoring function to use for conformal prediction. Options include:
            - "APS": Adaptive Prediction Sets.
            - "RAPS": Regularized Adaptive Prediction Sets.
            - "SAPS": Smoothed Adaptive Prediction Sets.
            - "THR": Threshold-based Prediction Sets.
            Default is "APS".
        alpha : float, optional
            The miscoverage level for conformal prediction. Determines the confidence level
            (e.g., alpha=0.1 corresponds to 90% confidence). Default is 0.1.
        kreg : int, optional
            Regularization parameter used in certain scoring functions like "RAPS" and "SAPS".
            Default is 1.

    Returns:
        prediction_sets : Tensor
            A tensor containing the prediction sets for the test samples. Each set contains
            the indices of the predicted classes for each test sample.

    Notes:
        - For "RAPS" and "SAPS" scoring functions, the calibration set is further split into
        a validation set (20%) and a calibration set (80%) to tune the hyperparameter `kreg`.
        - The function assumes that the calibration and test masks are disjoint.

    References:
        - "Conformal Prediction for Reliable Machine Learning: Theory and Applications" (2023)
        https://arxiv.org/pdf/2310.06430
    """

    if isinstance(calib_mask, np.ndarray):
        calib_mask = torch.tensor(calib_mask, dtype=bool)

    if isinstance(test_mask, np.ndarray):
        test_mask = torch.tensor(test_mask, dtype=bool)

    # Some scores require the choice of a hyperparameter
    # Following https://arxiv.org/pdf/2310.06430, split the calibration set to a
    #  calib_validation and calib_calibration set to choose the hyperparameter
    # The above authors do this with a 20:80 split
    initial_mask_size = torch.sum(calib_mask).item()
    if score_function in ["RAPS", "SAPS"]:

        # Split the calibration set into a calibration and validation set
        n_calib = torch.sum(calib_mask).item()
        n_valid = int(0.2 * n_calib)

        # Get indices of the calibration set
        calib_indices = np.where(calib_mask)[0]

        # Shuffle the indices
        np.random.shuffle(calib_indices)

        # Split into calibration and validation indices
        calib_valid_indices = calib_indices[:n_valid]
        calib_indices = calib_indices[n_valid:]

        # Create new masks
        calib_valid_mask = torch.zeros_like(calib_mask, dtype=bool)
        calib_valid_mask[calib_valid_indices] = True

        calib_mask = torch.zeros_like(calib_mask, dtype=bool)
        calib_mask[calib_indices] = True

        # Compute softmax probabilities
        n_calib = calib_mask.sum()
        n_calib_valid = calib_valid_mask.sum()
        smx = torch.nn.Softmax(dim=1)
        calib_heuristic = smx(output[calib_mask])
        calib_valid_heuristic = smx(output[calib_valid_mask])
        test_heuristic = smx(output[test_mask]).detach().numpy()

        assert (
            torch.sum(calib_mask).item() < initial_mask_size
        ), "Calibration mask not reduced"

    else:
        # Compute softmax probabilities
        n_calib = calib_mask.sum()
        smx = torch.nn.Softmax(dim=1)
        calib_heuristic = smx(output[calib_mask])  # .detach().numpy()
        test_heuristic = smx(output[test_mask]).detach().numpy()

    if score_function == "APS":
        calib_scores = (
            APS_scores(probs=calib_heuristic, label=data.y[calib_mask]).detach().numpy()
        )
    elif score_function == "THR":
        calib_scores = (
            THR_scores(probs=calib_heuristic, label=data.y[calib_mask]).detach().numpy()
        )
    elif score_function == "RAPS":

        pen_to_try = np.array([0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5])
        best_param = 0
        best_size = np.unique(data.y[calib_valid_mask]).shape[0]
        for pen in pen_to_try:

            calib_valid_scores = (
                RAPS_scores(
                    probs=calib_valid_heuristic,
                    label=data.y[calib_valid_mask],
                    penalty=pen,
                    kreg=kreg,
                )
                .detach()
                .numpy()
            )

            # Evaluate
            qhat = np.quantile(
                calib_valid_scores,
                np.ceil((n_calib_valid + 1) * (1 - alpha)) / n_calib_valid,
                method="higher",
            )

            test_pi = test_heuristic.argsort(1)[:, ::-1]
            test_srt = np.take_along_axis(test_heuristic, test_pi, axis=1).cumsum(
                axis=1
            )
            pred_sets = np.take_along_axis(
                test_srt <= qhat, test_pi.argsort(axis=1), axis=1
            )

            # Average size
            avg_size = np.mean(np.sum(pred_sets, axis=1))

            # print(f"Penalty: {pen}, Avg size: {avg_size}")
            if avg_size < best_size:
                best_param = pen
                best_size = avg_size

        # print(f"\nBest penalty: {best_param}")
        calib_scores = (
            RAPS_scores(
                probs=calib_heuristic,
                label=data.y[calib_mask],
                penalty=best_param,
                kreg=kreg,
            )
            .detach()
            .numpy()
        )
    elif score_function == "SAPS":
        wt_to_try = np.array(
            [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 2, 5]
        )
        best_param = wt_to_try[0]
        best_size = np.unique(data.y[calib_valid_mask]).shape[0]
        for wt in wt_to_try:

            calib_valid_scores = (
                SAPS_scores(
                    probs=calib_valid_heuristic,
                    label=data.y[calib_valid_mask],
                    weight=wt,
                )
                .detach()
                .numpy()
            )

            # Evaluate
            qhat = np.quantile(
                calib_valid_scores,
                np.ceil((n_calib_valid + 1) * (1 - alpha)) / n_calib_valid,
                method="higher",
            )

            test_pi = test_heuristic.argsort(1)[:, ::-1]
            test_srt = np.take_along_axis(test_heuristic, test_pi, axis=1).cumsum(
                axis=1
            )
            pred_sets = np.take_along_axis(
                test_srt <= qhat, test_pi.argsort(axis=1), axis=1
            )

            # Average size
            avg_size = np.mean(np.sum(pred_sets, axis=1))

            if avg_size < best_size:
                best_param = wt
                best_size = avg_size

        # print(f"\nBest weight: {best_param}")
        calib_scores = (
            SAPS_scores(
                probs=calib_heuristic,
                label=data.y[calib_mask],
                weight=best_param,
            )
            .detach()
            .numpy()
        )
    else:
        raise ValueError(f"Unknown method: {score_function}")

    # Get the score quantile
    qhat_quantile = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib

    if qhat_quantile > 1:
        raise ValueError(
            "Specified quantile is larger than 1. Either increase the number of calibration data points or increase alpha."
        )

    qhat = np.quantile(calib_scores, qhat_quantile, method="higher")
    test_pi = test_heuristic.argsort(1)[:, ::-1]
    test_srt = np.take_along_axis(test_heuristic, test_pi, axis=1).cumsum(axis=1)
    pred_sets = np.take_along_axis(test_srt <= qhat, test_pi.argsort(axis=1), axis=1)

    return pred_sets


def _sort_sum(probs):
    # ordered: the ordered probabilities in descending order
    # indices: the rank of ordered probabilities in descending order
    # cumsum: the accumulation of sorted probabilities
    ordered, indices = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(ordered, dim=-1)
    return indices, ordered, cumsum


def APS_scores(probs, label):

    indices, ordered, cumsum = _sort_sum(probs)
    U = torch.rand(indices.shape[0], device=probs.device)
    idx = torch.where(indices == label.view(-1, 1))
    # scores_first_rank = U * cumsum[idx]
    scores_first_rank = cumsum[idx]
    idx_minus_one = (idx[0], idx[1] - 1)
    # scores_usual = U * ordered[idx] + cumsum[idx_minus_one]
    scores_usual = ordered[idx] + cumsum[idx_minus_one]
    return torch.where(idx[1] == 0, scores_first_rank, scores_usual)


def RAPS_scores(probs, label, penalty, kreg):
    indices, ordered, cumsum = _sort_sum(probs)
    U = torch.rand(indices.shape[0], device=probs.device)
    idx = torch.where(indices == label.view(-1, 1))
    reg = torch.maximum(penalty * (idx[1] + 1 - kreg), torch.tensor(0).to(probs.device))
    # scores_first_rank = U * ordered[idx] + reg
    scores_first_rank = ordered[idx] + reg
    idx_minus_one = (idx[0], idx[1] - 1)
    # scores_usual = U * ordered[idx] + cumsum[idx_minus_one] + reg
    scores_usual = ordered[idx] + cumsum[idx_minus_one] + reg
    return torch.where(idx[1] == 0, scores_first_rank, scores_usual)


def THR_scores(probs, label):

    return 1 - probs[torch.arange(probs.shape[0], device=probs.device), label]


def SAPS_scores(probs, label, weight):

    if weight <= 0:
        raise ValueError("The parameter 'weight' must be a positive value.")

    indices, ordered, cumsum = _sort_sum(probs)
    # U = torch.rand(indices.shape[0], device=probs.device)
    idx = torch.where(indices == label.view(-1, 1))
    # scores_first_rank = U * cumsum[idx]
    scores_first_rank = cumsum[idx]
    # scores_usual = weight * (idx[1] - U) + ordered[:, 0]
    scores_usual = weight * (idx[1]) + ordered[:, 0]
    return torch.where(idx[1] == 0, scores_first_rank, scores_usual)
