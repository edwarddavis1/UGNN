import numpy as np
from tqdm import tqdm
import copy
import torch
from data import get_school_data
from ugnn.networks import Dynamic_Network, Unfolded_Network
from ugnn.utils.masks import non_zero_degree_mask, mask_split, pad_unfolded_mask
from ugnn.gnns import GCN, train, valid


# Load data
As, node_labels = get_school_data()
T = len(As)
n = As[0].shape[0]
num_classes = len(np.unique(node_labels))

# Convert to a torch geometric dataset containing T graphs
dyn_network = Dynamic_Network(As, node_labels)

# Unfold the T graphs into a single graph
unf_network = Unfolded_Network(dyn_network)[0]

# Set up masks for the specified regime
# See https://arxiv.org/abs/2405.19230 for details on different regimes
regime = "semi-inductive"
data_mask = non_zero_degree_mask(As, n, T)
train_mask, valid_mask, _, test_mask = mask_split(
    data_mask, split_props=[0.5, 0.3, 0, 0.2], regime=regime
)

# Pad masks to include anchor nodes (required when unfolding)
train_mask = pad_unfolded_mask(train_mask, n)
valid_mask = pad_unfolded_mask(valid_mask, n)
test_mask = pad_unfolded_mask(test_mask, n)

# Train a UGCN
model = GCN(
    num_nodes=unf_network.num_nodes, num_channels=16, num_classes=num_classes, seed=123
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

max_valid_acc = 0
for epoch in tqdm(range(200)):
    _ = train(model, unf_network, train_mask, optimizer)
    valid_acc = valid(model, unf_network, valid_mask)

    if valid_acc > max_valid_acc:
        max_valid_acc = valid_acc
        best_model = copy.deepcopy(model)

test_acc = valid(model, unf_network, test_mask)
print(f"Test accuracy: {test_acc:0.3f}")
