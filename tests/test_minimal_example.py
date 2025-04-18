import numpy as np
import torch
from data import get_school_data
from ugnn.networks import Dynamic_Network, Unfolded_Network
from ugnn.utils.masks import non_zero_degree_mask, mask_split, pad_unfolded_mask
from ugnn.gnns import GCN, train, valid


def test_minimal_example():
    # Mock or load data
    As, node_labels = get_school_data()
    n = As.shape[1]
    T = As.shape[0]
    num_classes = len(np.unique(node_labels))

    # Convert to a torch geometric dataset containing T graphs
    dyn_network = Dynamic_Network(As, node_labels)

    # Unfold the T graphs into a single graph
    unf_network = Unfolded_Network(dyn_network)[0]

    # Set up masks for the specified regime
    regime = "semi-inductive"
    data_mask = non_zero_degree_mask(As, n, T)
    train_mask, valid_mask, _, test_mask = mask_split(
        data_mask, split_props=[0.5, 0.3, 0, 0.2], regime=regime
    )
    train_mask = pad_unfolded_mask(train_mask, n)
    valid_mask = pad_unfolded_mask(valid_mask, n)

    # Train a UGCN
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
