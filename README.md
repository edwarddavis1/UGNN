# [IN DEVELOPMENT] UGNN: The Unfolded Graph Neural Networks package.

Welcome to the documentation for **UGNN**, a library for using the **unfolded graph neural network** (UGNN) model for discrete-time dynamic graphs.

For more details on this model, see the paper: [Valid Conformal Prediction for Dynamic GNNs](https://arxiv.org/abs/2405.19230), accepted at ICLR 2025.

## Installation

Package requires Python 3.11 or later. Once cloned, dependencies can be installed using the following command in the "UGNN" root directory.

```bash
pip install -e .
```

## Usage Example

Here is a minimal example of how to train an unfolded GCN (UGCN) model using the UGNN library.

```python
import numpy as np
import torch
from data.school.school_processing import get_school_data
from ugnn.networks import Dynamic_Network, Unfolded_Network
from ugnn.gnns import GCN, train, valid
from ugnn.utils.masks import non_zero_degree_mask, mask_split, pad_unfolded_mask

# Load example data - T adjacency matrices and n node labels
As, spatial_node_labels = get_school_data()
n = As.shape[1]
T = As.shape[0]
node_labels = np.tile(spatial_node_labels, T)
num_classes = len(np.unique(node_labels))

# Convert to a torch geometric dataset containing T graphs
dyn_network = Dynamic_Network(As, node_labels)

# "Unfold" the T graph dynamic network into a single graph
unf_network = Unfolded_Network(dyn_network)[0]

# Create masks for training and validation
data_mask = non_zero_degree_mask(As, As.shape[1], As.shape[0])
train_mask, valid_mask, _, test_mask = mask_split(
    data_mask, split_props=[0.5, 0.3, 0, 0.2], regime="semi-inductive"
)
train_mask = pad_unfolded_mask(train_mask, As.shape[1])
valid_mask = pad_unfolded_mask(valid_mask, As.shape[1])

# Train a UGCN
model = GCN(
    num_nodes=unf_network.num_nodes, num_channels=16, num_classes=num_classes, seed=123
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(10):  # Reduced epochs for brevity
    train(model, unf_network, train_mask, optimizer)
    valid_acc = valid(model, unf_network, valid_mask)
    print(f"Epoch {epoch}, Validation Accuracy: {valid_acc:.3f}")
```
