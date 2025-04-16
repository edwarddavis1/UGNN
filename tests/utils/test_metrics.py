import numpy as np
import torch
from torch_geometric.data import Data

from ugnn.utils.metrics import accuracy, avg_set_size, coverage


def test_accuracy():

    # Example where the output should predict 1, 1, 0
    # True labels are 1, 0, 1
    # Given that the last is masked out, accuracy should be 0.5

    output = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.6, 0.4]])
    data = Data(
        x=None,
        edge_index=None,
        y=torch.tensor([1, 0, 1]),
        test_mask=torch.tensor([True, True, False]),
    )
    acc = accuracy(output, data, data.test_mask)

    assert acc == 0.5  # Correct accuracy
    assert isinstance(acc, float)  # Correct type
    assert acc >= 0 and acc <= 1  # Correct range


def test_avg_set_size():
    # Example where 2/3 of the first set is True and 1/3 of the second set is True
    # The average size should be (2 + 1) / 2 = 1.5

    pred_sets = np.array([[True, False, True], [False, True, False]])
    avg_size = avg_set_size(pred_sets)

    assert avg_size == 1.5  # Correct average size
    assert isinstance(avg_size, float)  # Correct type
    assert avg_size >= 0  # Correct range


def test_coverage():

    # Example where the correct labels are 1, 0 for included test points
    # The first set is not covered, the second set is
    # Overall coverage should then be 0.5

    pred_sets = np.array(
        [[True, False, True], [True, False, False], [False, True, False]]
    )
    data = Data(
        x=None,
        edge_index=None,
        y=torch.tensor([1, 0, 0]),
        test_mask=torch.tensor([True, True, False]),
    )
    cov = coverage(pred_sets, data, data.test_mask)

    assert cov == 0.5  # Correct coverage
    assert isinstance(cov, float)  # Correct type
    assert cov >= 0 and cov <= 1  # Correct range
