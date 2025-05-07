# UGNN: The Unfolded Graph Neural Networks package.

Welcome to the documentation for **UGNN**, a library for using the **unfolded graph neural network**.

_UGNN is a powerful and interpretable model for embedding a collection of networks with a common node set._

For more details on this model, see the paper: [Valid Conformal Prediction for Dynamic GNNs](https://arxiv.org/abs/2405.19230), accepted at ICLR 2025.

## About Unfolded GNN

#### Definition

Let $\mathbf{A}^{(1)},\dots,\mathbf{A}^{(T)}$ be a collection of $T$ $n \times n$ adjacency matrices, each representing an $n$-node network. An **unfolding** of this collection is given as

```math
\begin{equation}
\mathbf{A} = \begin{pmatrix}
\mathbf{0} & \mathcal{A} \\ \mathcal{A}^\top & \mathbf{0}
\end{pmatrix},
\end{equation}
```

where $\mathbf{\mathcal{A}} = [\mathbf{A}^{(1)}, \dots, \mathbf{A}^{(T)}]$ is an $n \times nT$ column-concatenation of all networks in the set. _A UGNN is simply a GNN which takes an unfolded matrix as input._

#### Perks of UGNN

-   **Accuracy**: For tasks predicting node labels into the future, UGNN displays considerable gains over the PyTorch geometric established method (e.g. accuracy gains up to 92% vs 12%).
-   **Uncertainty Quantification**: UGNN allows for the application of _conformal prediction_ to quantify uncertainty on the prediction of future nodes.

## Installation

Package requires Python 3.11 or later. Once cloned, dependencies can be installed using the following command in the "UGNN" root directory.

```bash
pip install -e .
```

## Usage Example

Here is a minimal example of how to train an unfolded GCN (UGCN) model using the UGNN library. A notebook with a full example of UGNN training and conformal prediction is supplied in the `examples` directory.

```python
import numpy as np
import torch
from data import get_school_data
from ugnn.networks import Dynamic_Network, Unfolded_Network
from ugnn.gnns import GCN, train, valid
from ugnn.utils.masks import non_zero_degree_mask, mask_split, pad_unfolded_mask

# Load example data - T adjacency matrices and n node labels
As, node_labels = get_school_data()
T = len(As)
n = As[0].shape[0]
num_classes = len(np.unique(node_labels))

# Convert to a torch geometric dataset containing T graphs
dyn_network = Dynamic_Network(As, node_labels)

# "Unfold" the T graph dynamic network into a single graph
unf_network = Unfolded_Network(dyn_network)[0]

# Create masks for train/valid/calib/test for a selected regime
# Calib data only required if using conformal prediction downstream of training
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

for epoch in range(10):
    train(model, unf_network, train_mask, optimizer)
    valid_acc = valid(model, unf_network, valid_mask)
    print(f"Epoch {epoch}, Validation Accuracy: {valid_acc:.3f}")
```
