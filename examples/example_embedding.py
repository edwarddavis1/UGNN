# %%
import numpy as np
import copy
import torch
from data.school.school_processing import get_school_data
from ugnn.networks import Dynamic_Network, Unfolded_Network, Block_Diagonal_Network
from ugnn.utils.masks import non_zero_degree_mask, mask_split, pad_unfolded_mask
from ugnn.gnns import GCN, train, valid

# %%
regime = "semi-inductive"
# %%
# Load data
As, spatial_node_labels = get_school_data()

n = As.shape[1]
T = As.shape[0]
node_labels = np.tile(spatial_node_labels, T)
num_classes = len(np.unique(node_labels))

dyn_network = Dynamic_Network(As, node_labels)
# %%
# Unfold the network
unf_network = Unfolded_Network(dyn_network)[0]
# %%
# Set up masks
data_mask = non_zero_degree_mask(As, n, T)

# Define train/valid/calib/test split proportions
train_mask, valid_mask, _, test_mask = mask_split(
    data_mask, split_props=[0.5, 0.3, 0, 0.2], regime=regime
)

# Pad masks to include anchor nodes (required when unfolding)
train_mask = pad_unfolded_mask(train_mask, n)
valid_mask = pad_unfolded_mask(valid_mask, n)
test_mask = pad_unfolded_mask(test_mask, n)

# %%
model = GCN(
    num_nodes=unf_network.num_nodes, num_channels=16, num_classes=num_classes, seed=123
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

max_valid_acc = 0
for epoch in range(200):
    _ = train(model, unf_network, train_mask, optimizer)
    valid_acc = valid(model, unf_network, valid_mask)

    if valid_acc > max_valid_acc:
        max_valid_acc = valid_acc
        best_model = copy.deepcopy(model)

test_acc = valid(model, unf_network, test_mask)
print(f"Test accuracy: {test_acc:0.3f}")
