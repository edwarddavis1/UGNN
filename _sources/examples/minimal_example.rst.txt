Minimal Example
===============

Here is a minimal example of how to train an unfolded GCN (UGCN) model using the UGNN library. 

.. code-block:: python

    import numpy as np
    from ugnn.networks import Dynamic_Network, Unfolded_Network
    from ugnn.gnns import GCN, train, valid
    from ugnn.utils.masks import non_zero_degree_mask, mask_split, pad_unfolded_mask
    import torch

    # Load example data
    As = np.random.rand(10, 100, 100)  # Example adjacency matrices (T=10, n=100)
    node_labels = np.random.randint(0, 5, size=(100 * 10))  # Example node labels
    num_classes = len(np.unique(node_labels))

    # Convert to a dynamic network
    dyn_network = Dynamic_Network(As, node_labels)

    # Unfold the dynamic network into a single graph
    unf_network = Unfolded_Network(dyn_network)[0]

    # Create masks for training and validation
    data_mask = non_zero_degree_mask(As, As.shape[1], As.shape[0])
    train_mask, valid_mask, _, test_mask = mask_split(
        data_mask, split_props=[0.5, 0.3, 0, 0.2], regime="semi-inductive"
    )
    train_mask = pad_unfolded_mask(train_mask, As.shape[1])
    valid_mask = pad_unfolded_mask(valid_mask, As.shape[1])

    # Train a GCN model
    model = GCN(
        num_nodes=unf_network.num_nodes, num_channels=16, num_classes=num_classes, seed=123
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(10):  # Reduced epochs for brevity
        train(model, unf_network, train_mask, optimizer)
        valid_acc = valid(model, unf_network, valid_mask)
        print(f"Epoch {epoch}, Validation Accuracy: {valid_acc:.3f}")
