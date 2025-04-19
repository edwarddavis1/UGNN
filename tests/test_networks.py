import numpy as np
from scipy import sparse
from ugnn.networks import unfolded_matrix_from_series, block_diagonal_matrix_from_series


def test_unfolded_matrix_from_series():
    # Generate random data
    T = 5
    n = 50
    As = np.random.rand(T, n, n)

    unfolded_matrix = unfolded_matrix_from_series(As)

    # Check the shape of the unfolded matrix
    assert unfolded_matrix.shape == (
        n * (T + 1),
        n * (T + 1),
    ), "Unfolded matrix shape is incorrect"

    # Check that the unfolded matrix is symmetric
    assert np.allclose(
        unfolded_matrix, unfolded_matrix.T
    ), "Unfolded matrix is not symmetric"


def test_unfolded_matrix_from_series_sparse():
    # Generate random sparse data
    T = 5
    n = 50
    density = 0.1
    As = np.array(
        [
            sparse.random(n, n, density=density, format="coo", random_state=42)
            for _ in range(T)
        ]
    )

    unfolded_matrix = unfolded_matrix_from_series(As)

    # Check the shape of the unfolded matrix
    assert unfolded_matrix.shape == (
        n * (T + 1),
        n * (T + 1),
    ), "Unfolded matrix shape is incorrect"

    # Check output is sparse
    assert sparse.issparse(unfolded_matrix), "Unfolded matrix is not sparse"

    # Check that the unfolded matrix is symmetric
    assert (
        unfolded_matrix != unfolded_matrix.T
    ).nnz == 0, "Unfolded matrix is not symmetric"


def test_block_diagonal_matrix_from_series():
    # Generate random data
    T = 5
    n = 50
    As = np.random.rand(T, n, n)

    block_diag_matrix = block_diagonal_matrix_from_series(As)

    # Check the shape of the block diagonal matrix
    assert block_diag_matrix.shape == (
        n * T,
        n * T,
    ), "Block diagonal matrix shape is incorrect"

    # Check that the block diagonal matrix is block diagonal
    for i in range(T):
        assert np.allclose(
            block_diag_matrix[i * n : (i + 1) * n, i * n : (i + 1) * n],
            As[i],
        ), f"Block {i} of block diagonal matrix is incorrect"


def test_block_diagonal_matrix_from_series_sparse():
    # Generate random sparse data
    T = 5
    n = 50
    density = 0.1
    As = np.array(
        [
            sparse.random(n, n, density=density, format="coo", random_state=42)
            for _ in range(T)
        ]
    )

    block_diag_matrix = block_diagonal_matrix_from_series(As)

    # Check the shape of the block diagonal matrix
    assert block_diag_matrix.shape == (
        n * T,
        n * T,
    ), "Block diagonal matrix shape is incorrect"

    # Check output is sparse
    assert sparse.issparse(block_diag_matrix), "Block diagonal matrix is not sparse"
