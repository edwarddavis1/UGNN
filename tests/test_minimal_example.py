import numpy as np
import torch
from scipy.sparse import random as sparse_random
from ugnn.networks import Dynamic_Network, Unfolded_Network, Block_Diagonal_Network
from ugnn.utils.masks import non_zero_degree_mask, mask_split, pad_unfolded_mask
from ugnn.gnns import GCN, train, valid


def test_minimal_unfolded_example_with_random_data():
    # Generate random data
    T = 5
    n = 50
    num_classes = 3
    As = np.random.rand(T, n, n)
    node_labels = np.random.randint(0, num_classes, size=n * T)

    dyn_network = Dynamic_Network(As, node_labels)
    unf_network = Unfolded_Network(dyn_network)[0]

    regime = "semi-inductive"
    data_mask = non_zero_degree_mask(As, n, T)
    train_mask, valid_mask, _, _ = mask_split(
        data_mask, split_props=[0.2, 0.1, 0.35, 0.35], regime=regime
    )
    train_mask = pad_unfolded_mask(train_mask, n)
    valid_mask = pad_unfolded_mask(valid_mask, n)

    model = GCN(
        num_nodes=unf_network.num_nodes,
        num_channels=16,
        num_classes=num_classes,
        seed=123,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    _ = train(model, unf_network, train_mask, optimizer)
    valid_acc = valid(model, unf_network, valid_mask)

    assert valid_acc >= 0, "Validation accuracy should be non-negative"


def test_minimal_unfolded_example_with_sparse_data():
    # Generate random sparse data
    T = 5
    n = 50
    num_classes = 3
    density = 0.1
    As = np.array(
        [
            sparse_random(
                n, n, density=density, format="csr", random_state=42
            ).toarray()
            for _ in range(T)
        ]
    )
    node_labels = np.random.randint(0, num_classes, size=n * T)

    dyn_network = Dynamic_Network(As, node_labels)
    unf_network = Unfolded_Network(dyn_network)[0]

    regime = "semi-inductive"
    data_mask = non_zero_degree_mask(As, n, T)
    train_mask, valid_mask, _, _ = mask_split(
        data_mask, split_props=[0.2, 0.1, 0.35, 0.35], regime=regime
    )
    train_mask = pad_unfolded_mask(train_mask, n)
    valid_mask = pad_unfolded_mask(valid_mask, n)

    model = GCN(
        num_nodes=unf_network.num_nodes,
        num_channels=16,
        num_classes=num_classes,
        seed=123,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    _ = train(model, unf_network, train_mask, optimizer)
    valid_acc = valid(model, unf_network, valid_mask)

    assert valid_acc >= 0, "Validation accuracy should be non-negative"


def test_minimal_block_diagonal_example_with_random_data():
    # Generate random data
    T = 5
    n = 50
    num_classes = 3
    As = np.random.rand(T, n, n)
    node_labels = np.random.randint(0, num_classes, size=n * T)

    dyn_network = Dynamic_Network(As, node_labels)
    block_diag_network = Block_Diagonal_Network(dyn_network)[0]

    regime = "semi-inductive"
    data_mask = non_zero_degree_mask(As, n, T)
    train_mask, valid_mask, _, _ = mask_split(
        data_mask, split_props=[0.2, 0.1, 0.35, 0.35], regime=regime
    )

    model = GCN(
        num_nodes=block_diag_network.num_nodes,
        num_channels=16,
        num_classes=num_classes,
        seed=123,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    _ = train(model, block_diag_network, train_mask, optimizer)
    valid_acc = valid(model, block_diag_network, valid_mask)

    assert valid_acc >= 0, "Validation accuracy should be non-negative"


def test_minimal_block_diagonal_example_with_sparse_data():
    # Generate random sparse data
    T = 5
    n = 50
    num_classes = 3
    density = 0.1
    As = np.array(
        [
            sparse_random(
                n, n, density=density, format="csr", random_state=42
            ).toarray()
            for _ in range(T)
        ]
    )
    node_labels = np.random.randint(0, num_classes, size=n * T)

    dyn_network = Dynamic_Network(As, node_labels)
    block_diag_network = Block_Diagonal_Network(dyn_network)[0]

    regime = "semi-inductive"
    data_mask = non_zero_degree_mask(As, n, T)
    train_mask, valid_mask, _, _ = mask_split(
        data_mask, split_props=[0.2, 0.1, 0.35, 0.35], regime=regime
    )

    model = GCN(
        num_nodes=block_diag_network.num_nodes,
        num_channels=16,
        num_classes=num_classes,
        seed=123,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    _ = train(model, block_diag_network, train_mask, optimizer)
    valid_acc = valid(model, block_diag_network, valid_mask)

    assert valid_acc >= 0, "Validation accuracy should be non-negative"
