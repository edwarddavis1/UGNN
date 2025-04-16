import numpy as np
from torch import Tensor
from torch_geometric.data import Data


def accuracy(output: Tensor, data: Data, test_mask: Tensor) -> float:
    """
    Calculate the accuracy of predictions.

    Args:
        output (Tensor): Model output logits.
        data (Data): Graph data containing ground truth labels.
        test_mask (Tensor): Mask indicating test nodes.

    Returns:
        float: Accuracy of the predictions.
    """
    pred = output.argmax(dim=1)
    correct = pred[test_mask] == data.y[test_mask]
    acc = int(correct.sum()) / int(test_mask.sum())
    return acc


def avg_set_size(pred_sets: np.ndarray) -> float:
    """
    Calculate the average size of prediction sets.

    Args:
        pred_sets (np.ndarray): Array of prediction sets.
        test_mask (np.ndarray): Mask indicating test nodes.

    Returns:
        float: Average size of prediction sets.
    """
    return np.mean(np.sum(pred_sets, axis=1))


def coverage(
    pred_sets: np.ndarray[np.bool_], data: Data, test_mask: np.ndarray
) -> float:
    """
    Calculate the coverage of prediction sets.

    Args:
        pred_sets (np.ndarray): Array of prediction sets.
        data (Data): Graph data containing ground truth labels.
        test_mask (np.ndarray): Mask indicating test nodes.

    Returns:
        float: Coverage of the prediction sets.
    """
    in_set = np.array(
        [pred_set[label] for pred_set, label in zip(pred_sets, data.y[test_mask])]
    )
    return np.mean(in_set)
