# %%
import copy
from itertools import product
import numpy as np
import pickle
import pyemb as eb
from tqdm import tqdm

import torch

from ugnn.networks import Dynamic_Network, Block_Diagonal_Network, Unfolded_Network
from ugnn.gnns import GCN, GAT, train, valid
from ugnn.utils import accuracy, avg_set_size, coverage
from ugnn.config import EXPERIMENT_PARAMS

# %%
# Unpack parameters from EXPERIMENT_PARAMS
props = EXPERIMENT_PARAMS["props"]
alpha = EXPERIMENT_PARAMS["alpha"]
num_train_trans = EXPERIMENT_PARAMS["num_train_trans"]
num_permute_trans = EXPERIMENT_PARAMS["num_permute_trans"]
num_train_semi_ind = EXPERIMENT_PARAMS["num_train_semi_ind"]
num_epochs = EXPERIMENT_PARAMS["num_epochs"]
num_channels_GCN = EXPERIMENT_PARAMS["num_channels_GCN"]
num_channels_GAT = EXPERIMENT_PARAMS["num_channels_GAT"]
learning_rate = EXPERIMENT_PARAMS["learning_rate"]
weight_decay = EXPERIMENT_PARAMS["weight_decay"]


# Save results
results_file = f"results/SBM_with_less_test.pkl"

# %% [markdown]
# ## Generate dataset

# %%
np.random.seed(42)

# %% [markdown]
# Set dynamic SBM parameters.

# %%
K = 3
n = 100 * K
T = 8
pi = np.repeat(1 / K, K)

a = [0.08, 0.16]
Bs = 0.02 * np.ones((T, K, K))

T_list = [t for t in range(T)]
np.random.shuffle(T_list)

for t in range(T):
    for k in range(K):
        Bs[t, k, k] = a[(T_list[t] & (1 << k)) >> k]

# %%
# As, Z = se.generate_SBM_dynamic(n, Bs, pi)
As, Z = eb.simulation.SBM(n, Bs, pi)

node_labels = np.tile(Z, T)
num_classes = K

# %%
# # (For the dataset plotting script - not required for the experiments to run)

# from scipy import sparse

# As_sparse = [sparse.csr_matrix(A) for A in As]
# for i, A_sparse in enumerate(As_sparse):
#     sparse.save_npz(f"data/sbm/sbm_As_{i}.npz", A_sparse)

# np.save(f"data/sbm/sbm_node_labels.npy", node_labels)

# %% [markdown]
# ## GNN functions

# %% [markdown]
# Dynamic network class that puts a list of adjacency matrices along with class labels into a pytorch geometric dataset. This is then used to create the block diagonal and dilated unfolded adjacency matrices for input into the graph neural networks.

# %%
dataset = Dynamic_Network(As, node_labels)
dataset_BD = Block_Diagonal_Network(dataset)[0]
dataset_UA = Unfolded_Network(dataset)[0]

# %% [markdown]
# Define general GCN and GAT neural networks to be applied to both the block diagonal and dilated unfolded networks.

# %% [markdown]
# ## Data split functions

# %% [markdown]
# Create a data mask that consists only of useful nodes for training and testing. We only consider nodes in a particular time window if it has degree greater than zero at that time window.

# %%
data_mask = np.array([[True] * T for _ in range(n)])

for t in range(T):
    data_mask[np.where(np.sum(As[t], axis=0) == 0)[0], t] = False

print(
    f"Percentage of usable node/time pairs: {100 * np.sum(data_mask) / (n * T) :02.1f}%"
)


# %%
def mask_split(mask, split_props, seed=0, mode="transductive"):
    np.random.seed(seed)

    n, T = mask.shape

    if mode == "transductive":
        # Flatten mask array into one dimension in blocks of nodes per time
        flat_mask = mask.T.reshape(-1)
        n_masks = np.sum(flat_mask)

        # Split shuffled flatten mask array indices into correct proportions
        flat_mask_idx = np.where(flat_mask)[0]
        np.random.shuffle(flat_mask_idx)
        split_ns = np.cumsum([round(n_masks * prop) for prop in split_props[:-1]])
        split_idx = np.split(flat_mask_idx, split_ns)

    if mode == "semi-inductive":

        # Find time such that final proportion of masks happen after that time
        T_trunc = np.where(
            np.cumsum(np.sum(mask, axis=0) / np.sum(mask)) >= 1 - split_props[-1]
        )[0][0]

        # Flatten mask arrays into one dimension in blocks of nodes per time
        flat_mask_start = mask[:, :T_trunc].T.reshape(-1)
        flat_mask_end = mask[:, T_trunc:].T.reshape(-1)
        n_masks_start = np.sum(flat_mask_start)

        # Split starting shuffled flatten mask array into correct proportions
        flat_mask_start_idx = np.where(flat_mask_start)[0]
        np.random.shuffle(flat_mask_start_idx)
        split_props_start = split_props[:-1] / np.sum(split_props[:-1])
        split_ns = np.cumsum(
            [round(n_masks_start * prop) for prop in split_props_start[:-1]]
        )
        split_idx = np.split(flat_mask_start_idx, split_ns)

        # Place finishing flatten mask array at the end
        split_idx.append(n * T_trunc + np.where(flat_mask_end)[0])

    if mode == "assisted semi-inductive":

        # Find time such that final proportion of masks happen after that time
        T_trunc = np.where(
            np.cumsum(np.sum(mask, axis=0) / np.sum(mask)) >= 1 - split_props[-1]
        )[0][0]

        # Flatten mask arrays into one dimension in blocks of nodes per time
        flat_mask_start = mask[:, :T_trunc].T.reshape(-1)
        flat_mask_end = mask[:, T_trunc:].T.reshape(-1)
        n_masks_start = np.sum(flat_mask_start)
        n_masks_end = np.sum(flat_mask_end)

        # Split starting shuffled flatten mask array into correct proportions
        flat_mask_start_idx = np.where(flat_mask_start)[0]
        np.random.shuffle(flat_mask_start_idx)
        split_props_start = split_props[:-2] / np.sum(split_props[:-2])
        split_ns = np.cumsum(
            [round(n_masks_start * prop) for prop in split_props_start[:-1]]
        )
        split_idx = np.split(flat_mask_start_idx, split_ns)

        # Do the same for after T_trunc
        flat_mask_end_idx = np.where(flat_mask_end)[0]
        np.random.shuffle(flat_mask_end_idx)
        split_props_end = split_props[-2:] / np.sum(split_props[-2:])
        split_ns = np.cumsum(
            [round(n_masks_end * prop) for prop in split_props_end[:-1]]
        )
        split_idx.append(n * T_trunc + np.split(flat_mask_end_idx, split_ns)[0])
        split_idx.append(n * T_trunc + np.split(flat_mask_end_idx, split_ns)[1])

    split_masks = np.array([[False] * n * T for _ in range(len(split_props))])
    for i in range(len(split_props)):
        split_masks[i, split_idx[i]] = True

    return split_masks


# %%
def mask_mix(mask_1, mask_2, seed=0):
    np.random.seed(seed)

    n = len(mask_1)
    n1 = np.sum(mask_1)
    n2 = np.sum(mask_2)

    mask_idx = np.where(mask_1 + mask_2)[0]
    np.random.shuffle(mask_idx)
    split_idx = np.split(mask_idx, [n1])

    split_masks = np.array([[False] * n for _ in range(2)])
    for i in range(2):
        split_masks[i, split_idx[i]] = True

    return split_masks


# %% [markdown]
# Testing the training/validation/calibration/test data split functions.

# %%
train_mask, valid_mask, calib_mask, test_mask = mask_split(
    data_mask, props, mode="semi-inductive"
)

print(f"Percentage train: {100 * np.sum(train_mask) / np.sum(data_mask) :02.1f}%")
print(f"Percentage valid: {100 * np.sum(valid_mask) / np.sum(data_mask) :02.1f}%")
print(f"Percentage calib: {100 * np.sum(calib_mask) / np.sum(data_mask) :02.1f}%")
print(f"Percentage test:  {100 * np.sum(test_mask)  / np.sum(data_mask) :02.1f}%")

# %% [markdown]
# ## Conformal prediction functions


# %%
def get_prediction_sets(output, data, calib_mask, test_mask, alpha=0.1):
    n_calib = calib_mask.sum()

    # Compute softmax probabilities
    smx = torch.nn.Softmax(dim=1)
    calib_heuristic = smx(output[calib_mask]).detach().numpy()
    test_heuristic = smx(output[test_mask]).detach().numpy()

    # APS
    calib_pi = calib_heuristic.argsort(1)[:, ::-1]
    calib_srt = np.take_along_axis(calib_heuristic, calib_pi, axis=1).cumsum(axis=1)
    calib_scores = np.take_along_axis(calib_srt, calib_pi.argsort(axis=1), axis=1)[
        range(n_calib), data.y[calib_mask]
    ]

    # Get the score quantile
    qhat = np.quantile(
        calib_scores, np.ceil((n_calib + 1) * (1 - alpha)) / n_calib, method="higher"
    )

    test_pi = test_heuristic.argsort(1)[:, ::-1]
    test_srt = np.take_along_axis(test_heuristic, test_pi, axis=1).cumsum(axis=1)
    pred_sets = np.take_along_axis(test_srt <= qhat, test_pi.argsort(axis=1), axis=1)

    return pred_sets


# %% [markdown]
# ## GNN training

# %% [markdown]
# Initiate nested results data structure.

# %%
results = {}

methods = ["BD", "UA"]
GNN_models = ["GCN", "GAT"]
regimes = ["Assisted Semi-Ind", "Trans", "Semi-Ind"]
# regimes = ["Trans", "Semi-Ind"]
outputs = ["Accuracy", "Avg Size", "Coverage"]
times = ["All"] + list(range(T))

for method in methods:
    results[method] = {}

    for GNN_model in GNN_models:
        results[method][GNN_model] = {}

        for regime in regimes:
            results[method][GNN_model][regime] = {}

            for output in outputs:
                results[method][GNN_model][regime][output] = {}

                for time in times:
                    results[method][GNN_model][regime][output][time] = []

# %% [markdown]
# ### Assisted Semi-inductive experiments

# %%

for method, GNN_model in product(methods, GNN_models):

    for i in range(num_train_trans):
        # Split data into training/validation/calibration/test
        train_mask, valid_mask, calib_mask, test_mask = mask_split(
            data_mask, props, seed=i, mode="assisted semi-inductive"
        )

        if method == "BD":
            method_str = "Block Diagonal"
            data = dataset_BD
        if method == "UA":
            method_str = "Unfolded"
            data = dataset_UA
            # Pad masKs to include anchor nodes
            train_mask = np.concatenate((np.array([False] * n), train_mask))
            valid_mask = np.concatenate((np.array([False] * n), valid_mask))
            calib_mask = np.concatenate((np.array([False] * n), calib_mask))
            test_mask = np.concatenate((np.array([False] * n), test_mask))

        if GNN_model == "GCN":
            model = GCN(data.num_nodes, num_channels_GCN, num_classes, seed=i)
        if GNN_model == "GAT":
            model = GAT(data.num_nodes, num_channels_GAT, num_classes, seed=i)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        print(f"\nTraining {method_str} {GNN_model} Number {i}")
        max_valid_acc = 0

        for epoch in tqdm(range(num_epochs)):
            train_loss = train(model, data, train_mask, optimizer)
            valid_acc = valid(model, data, valid_mask)

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                best_model = copy.deepcopy(model)

        print(f"Evaluating {method_str} {GNN_model} Number {i}")
        print(f"Validation accuracy: {max_valid_acc:0.3f}")
        output = best_model(data.x, data.edge_index)

        for j in tqdm(range(num_permute_trans)):
            # Permute the calibration and test datasets
            calib_mask, test_mask = mask_mix(calib_mask, test_mask, seed=j)

            pred_sets = get_prediction_sets(output, data, calib_mask, test_mask, alpha)

            results[method][GNN_model]["Assisted Semi-Ind"]["Accuracy"]["All"].append(
                accuracy(output, data, test_mask)
            )
            results[method][GNN_model]["Assisted Semi-Ind"]["Avg Size"]["All"].append(
                avg_set_size(pred_sets)
            )
            coverage_value = coverage(pred_sets, data, test_mask)
            results[method][GNN_model]["Assisted Semi-Ind"]["Coverage"]["All"].append(
                coverage_value
            )
            print(f"Coverage: {coverage_value:0.3f}")

            for t in range(T):
                # Consider test nodes only at time t
                if method == "BD":
                    time_mask = np.array([[False] * n for _ in range(T)])
                    time_mask[t] = True
                    time_mask = time_mask.reshape(-1)
                if method == "UA":
                    time_mask = np.array([[False] * n for _ in range(T + 1)])
                    time_mask[t + 1] = True
                    time_mask = time_mask.reshape(-1)

                test_mask_t = time_mask * test_mask
                if np.sum(test_mask_t) == 0:
                    continue

                # Get prediction sets corresponding to time t
                pred_sets_t = pred_sets[
                    np.array(
                        [
                            np.where(
                                np.where(test_mask)[0] == np.where(test_mask_t)[0][i]
                            )[0][0]
                            for i in range(sum(test_mask_t))
                        ]
                    )
                ]

                results[method][GNN_model]["Assisted Semi-Ind"]["Accuracy"][t].append(
                    accuracy(output, data, test_mask_t)
                )
                results[method][GNN_model]["Assisted Semi-Ind"]["Avg Size"][t].append(
                    avg_set_size(pred_sets_t)
                )
                results[method][GNN_model]["Assisted Semi-Ind"]["Coverage"][t].append(
                    coverage(pred_sets_t, data, test_mask_t)
                )

        avg_test_acc = np.mean(
            results[method][GNN_model]["Assisted Semi-Ind"]["Accuracy"]["All"]
        )
        print(f"Test accuracy: {avg_test_acc:0.3f}")


# %% [markdown]
# ### Transductive experiments

# %%
dataset[0]

# %%


for method, GNN_model in product(methods, GNN_models):

    for i in range(num_train_trans):
        # Split data into training/validation/calibration/test
        train_mask, valid_mask, calib_mask, test_mask = mask_split(
            data_mask, props, seed=i, mode="transductive"
        )

        if method == "BD":
            method_str = "Block Diagonal"
            data = dataset_BD
        if method == "UA":
            method_str = "Unfolded"
            data = dataset_UA
            # Pad masks to include anchor nodes
            train_mask = np.concatenate((np.array([False] * n), train_mask))
            valid_mask = np.concatenate((np.array([False] * n), valid_mask))
            calib_mask = np.concatenate((np.array([False] * n), calib_mask))
            test_mask = np.concatenate((np.array([False] * n), test_mask))

        if GNN_model == "GCN":
            model = GCN(data.num_nodes, num_channels_GCN, num_classes, seed=i)
        if GNN_model == "GAT":
            model = GAT(data.num_nodes, num_channels_GAT, num_classes, seed=i)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        print(f"\nTraining {method_str} {GNN_model} Number {i}")
        max_valid_acc = 0

        for epoch in tqdm(range(num_epochs)):
            train_loss = train(model, data, train_mask, optimizer)
            valid_acc = valid(model, data, valid_mask)

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                best_model = copy.deepcopy(model)

        print(f"Validation accuracy: {max_valid_acc:0.3f}")
        print(f"Evaluating {method_str} {GNN_model} Number {i}")
        output = best_model(data.x, data.edge_index)

        for j in tqdm(range(num_permute_trans)):
            # Permute the calibration and test datasets
            calib_mask, test_mask = mask_mix(calib_mask, test_mask, seed=j)

            pred_sets = get_prediction_sets(output, data, calib_mask, test_mask, alpha)

            results[method][GNN_model]["Trans"]["Accuracy"]["All"].append(
                accuracy(output, data, test_mask)
            )
            results[method][GNN_model]["Trans"]["Avg Size"]["All"].append(
                avg_set_size(pred_sets)
            )
            coverage_value = coverage(pred_sets, data, test_mask)
            results[method][GNN_model]["Trans"]["Coverage"]["All"].append(
                coverage_value
            )
            print(f"Coverage: {coverage_value:0.3f}")

            for t in range(T):
                # Consider test nodes only at time t
                if method == "BD":
                    time_mask = np.array([[False] * n for _ in range(T)])
                    time_mask[t] = True
                    time_mask = time_mask.reshape(-1)
                if method == "UA":
                    time_mask = np.array([[False] * n for _ in range(T + 1)])
                    time_mask[t + 1] = True
                    time_mask = time_mask.reshape(-1)

                test_mask_t = time_mask * test_mask
                if np.sum(test_mask_t) == 0:
                    continue

                # Get prediction sets corresponding to time t
                pred_sets_t = pred_sets[
                    np.array(
                        [
                            np.where(
                                np.where(test_mask)[0] == np.where(test_mask_t)[0][i]
                            )[0][0]
                            for i in range(sum(test_mask_t))
                        ]
                    )
                ]

                results[method][GNN_model]["Trans"]["Accuracy"][t].append(
                    accuracy(output, data, test_mask_t)
                )
                results[method][GNN_model]["Trans"]["Avg Size"][t].append(
                    avg_set_size(pred_sets_t)
                )
                results[method][GNN_model]["Trans"]["Coverage"][t].append(
                    coverage(pred_sets_t, data, test_mask_t)
                )

        avg_test_acc = np.mean(results[method][GNN_model]["Trans"]["Accuracy"]["All"])
        print(f"Test accuracy: {avg_test_acc:0.3f}")

# %% [markdown]
# ### Semi-inductive experiments

# %%

for method, GNN_model in product(methods, GNN_models):

    for i in range(num_train_semi_ind):
        # Split data into training/validation/calibration/test
        train_mask, valid_mask, calib_mask, test_mask = mask_split(
            data_mask, props, seed=i, mode="semi-inductive"
        )

        if method == "BD":
            method_str = "Block Diagonal"
            data = dataset_BD
        if method == "UA":
            method_str = "Unfolded"
            data = dataset_UA
            # Pad masks to include anchor nodes
            train_mask = np.concatenate((np.array([False] * n), train_mask))
            valid_mask = np.concatenate((np.array([False] * n), valid_mask))
            calib_mask = np.concatenate((np.array([False] * n), calib_mask))
            test_mask = np.concatenate((np.array([False] * n), test_mask))

        if GNN_model == "GCN":
            model = GCN(data.num_nodes, num_channels_GCN, num_classes, seed=i)
        if GNN_model == "GAT":
            model = GAT(data.num_nodes, num_channels_GAT, num_classes, seed=i)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        print(f"\nTraining {method_str} {GNN_model} Number {i}")
        max_valid_acc = 0

        for epoch in tqdm(range(num_epochs)):
            train_loss = train(model, data, train_mask, optimizer)
            valid_acc = valid(model, data, valid_mask)

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                best_model = copy.deepcopy(model)

        print(f"Evaluating {method_str} {GNN_model} Number {i}")
        print(f"Validation accuracy: {max_valid_acc:0.3f}")
        output = best_model(data.x, data.edge_index)

        # Cannot permute the calibration and test datasets in semi-inductive experiments

        pred_sets = get_prediction_sets(output, data, calib_mask, test_mask, alpha)

        results[method][GNN_model]["Semi-Ind"]["Accuracy"]["All"].append(
            accuracy(output, data, test_mask)
        )
        results[method][GNN_model]["Semi-Ind"]["Avg Size"]["All"].append(
            avg_set_size(pred_sets)
        )
        coverage_value = coverage(pred_sets, data, test_mask)
        results[method][GNN_model]["Semi-Ind"]["Coverage"]["All"].append(coverage_value)
        print(f"Coverage: {coverage_value:0.3f}")

        for t in range(T):
            # Consider test nodes only at time t
            if method == "BD":
                time_mask = np.array([[False] * n for _ in range(T)])
                time_mask[t] = True
                time_mask = time_mask.reshape(-1)
            if method == "UA":
                time_mask = np.array([[False] * n for _ in range(T + 1)])
                time_mask[t + 1] = True
                time_mask = time_mask.reshape(-1)

            test_mask_t = time_mask * test_mask
            if np.sum(test_mask_t) == 0:
                continue

            # Get prediction sets corresponding to time t
            pred_sets_t = pred_sets[
                np.array(
                    [
                        np.where(np.where(test_mask)[0] == np.where(test_mask_t)[0][i])[
                            0
                        ][0]
                        for i in range(sum(test_mask_t))
                    ]
                )
            ]

            results[method][GNN_model]["Semi-Ind"]["Accuracy"][t].append(
                accuracy(output, data, test_mask_t)
            )
            results[method][GNN_model]["Semi-Ind"]["Avg Size"][t].append(
                avg_set_size(pred_sets_t)
            )
            results[method][GNN_model]["Semi-Ind"]["Coverage"][t].append(
                coverage(pred_sets_t, data, test_mask_t)
            )

        avg_test_acc = np.mean(
            results[method][GNN_model]["Semi-Ind"]["Accuracy"]["All"]
        )
        print(f"Test accuracy: {avg_test_acc:0.3f}")

# %% [markdown]
# Save results to pickle file.

# %%
with open(results_file, "wb") as file:
    pickle.dump(results, file)
