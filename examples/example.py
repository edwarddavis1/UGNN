# %%
import copy
from itertools import product
import numpy as np
import pickle
import pyemb as eb
from tqdm import tqdm
import os
from datetime import datetime

import torch

from ugnn.networks import Dynamic_Network, Block_Diagonal_Network, Unfolded_Network
from ugnn.gnns import GCN, GAT, train, valid
from ugnn.utils.metrics import accuracy, avg_set_size, coverage
from ugnn.config import EXPERIMENT_PARAMS
from ugnn.utils.masks import mask_split, mask_mix, non_zero_degree_mask
from ugnn.conformal import get_prediction_sets
from ugnn.experiments import Experiment
from ugnn.results_manager import ResultsManager

np.random.seed(42)
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

methods = EXPERIMENT_PARAMS["methods"]
GNN_models = EXPERIMENT_PARAMS["GNN_models"]
regimes = EXPERIMENT_PARAMS["regimes"]
outputs = EXPERIMENT_PARAMS["outputs"]


# Prepare results directory
experiment_name = "example_experiment"
results_dir = f"results/{experiment_name}"
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"{results_dir}/experiment_{timestamp}.pkl"

# %%
# GENERATE DATASET

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

As, Z = eb.simulation.SBM(n, Bs, pi)

node_labels = np.tile(Z, T)
num_classes = K

# %%
# Convert the data from adjacency matrices and labels to torch geometric datasets
# One for the block diagonal representation and one for the unfolded representation
dataset = Dynamic_Network(As, node_labels)
dataset_BD = Block_Diagonal_Network(dataset)[0]
dataset_UA = Unfolded_Network(dataset)[0]

# %%
DATA_PARAMS = {
    "n": n,
    "T": T,
    "num_classes": num_classes,
}

# %%
# Remove nodes with zero degree for each time point
data_mask = non_zero_degree_mask(As, n, T)

# Randomly separate remaining nodes according to data split proportions and regime
train_mask, valid_mask, calib_mask, test_mask = mask_split(
    data_mask, props, regime="semi-inductive"
)

# %%
results = {}

# methods = ["BD", "UA"]
# GNN_models = ["GCN", "GAT"]
# regimes = ["Assisted Semi-Ind", "Trans", "Semi-Ind"]
# # regimes = ["Trans", "Semi-Ind"]
# outputs = ["Accuracy", "Avg Size", "Coverage"]
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

# %%

regime = "semi-inductive"

# REMOVE THIS
method = methods[0]
GNN_model = GNN_models[0]
i = 0

res = ResultsManager(
    experiment_name=experiment_name,
)


for method, GNN_model in product(methods, GNN_models):

    for i in range(num_train_semi_ind):

        # Split data into training/validation/calibration/test
        train_mask, valid_mask, calib_mask, test_mask = mask_split(
            data_mask, props, seed=i, regime=regime
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

        # Initialise the experiment
        exp = Experiment(
            method=method,
            GNN_model=GNN_model,
            regime=regime,
            data=data,
            masks={
                "train": train_mask,
                "valid": valid_mask,
                "calib": calib_mask,
                "test": test_mask,
            },
            # Parameters for the experiment run (indep. of data)
            experiment_params=EXPERIMENT_PARAMS,
            # Parameters for the dataset (indep. of experiment)
            data_params=DATA_PARAMS,
        )

        # Train the GNN on the data
        exp.train()

        # Compute prediction sets and evaluate performance
        exp.evaluate()


res.save_results(exp.results)


# %%

# for method, GNN_model in product(methods, GNN_models):

#     for i in range(num_train_semi_ind):
#         # Split data into training/validation/calibration/test
#         train_mask, valid_mask, calib_mask, test_mask = mask_split(
#             data_mask, props, seed=i, regime="semi-inductive"
#         )

#         if method == "BD":
#             method_str = "Block Diagonal"
#             data = dataset_BD
#         if method == "UA":
#             method_str = "Unfolded"
#             data = dataset_UA
#             # Pad masks to include anchor nodes
#             train_mask = np.concatenate((np.array([False] * n), train_mask))
#             valid_mask = np.concatenate((np.array([False] * n), valid_mask))
#             calib_mask = np.concatenate((np.array([False] * n), calib_mask))
#             test_mask = np.concatenate((np.array([False] * n), test_mask))

#         if GNN_model == "GCN":
#             model = GCN(data.num_nodes, num_channels_GCN, num_classes, seed=i)
#         if GNN_model == "GAT":
#             model = GAT(data.num_nodes, num_channels_GAT, num_classes, seed=i)

#         optimizer = torch.optim.Adam(
#             model.parameters(), lr=learning_rate, weight_decay=weight_decay
#         )

#         print(f"\nTraining {method_str} {GNN_model} Number {i}")
#         max_valid_acc = 0

#         for epoch in tqdm(range(num_epochs)):
#             train_loss = train(model, data, train_mask, optimizer)
#             valid_acc = valid(model, data, valid_mask)

#             if valid_acc > max_valid_acc:
#                 max_valid_acc = valid_acc
#                 best_model = copy.deepcopy(model)

#         print(f"Evaluating {method_str} {GNN_model} Number {i}")
#         print(f"Validation accuracy: {max_valid_acc:0.3f}")
#         output = best_model(data.x, data.edge_index)

#         # Cannot permute the calibration and test datasets in semi-inductive experiments

#         pred_sets = get_prediction_sets(
#             output, data, calib_mask, test_mask, alpha, method="APS"
#         )

#         results[method][GNN_model]["Semi-Ind"]["Accuracy"]["All"].append(
#             accuracy(output, data, test_mask)
#         )
#         results[method][GNN_model]["Semi-Ind"]["Avg Size"]["All"].append(
#             avg_set_size(pred_sets)
#         )
#         coverage_value = coverage(pred_sets, data, test_mask)
#         results[method][GNN_model]["Semi-Ind"]["Coverage"]["All"].append(coverage_value)
#         print(f"Coverage: {coverage_value:0.3f}")

#         for t in range(T):
#             # Consider test nodes only at time t
#             if method == "BD":
#                 time_mask = np.array([[False] * n for _ in range(T)])
#                 time_mask[t] = True
#                 time_mask = time_mask.reshape(-1)
#             if method == "UA":
#                 time_mask = np.array([[False] * n for _ in range(T + 1)])
#                 time_mask[t + 1] = True
#                 time_mask = time_mask.reshape(-1)

#             test_mask_t = time_mask * test_mask
#             if np.sum(test_mask_t) == 0:
#                 continue

#             # Get prediction sets corresponding to time t
#             pred_sets_t = pred_sets[
#                 np.array(
#                     [
#                         np.where(np.where(test_mask)[0] == np.where(test_mask_t)[0][i])[
#                             0
#                         ][0]
#                         for i in range(sum(test_mask_t))
#                     ]
#                 )
#             ]

#             results[method][GNN_model]["Semi-Ind"]["Accuracy"][t].append(
#                 accuracy(output, data, test_mask_t)
#             )
#             results[method][GNN_model]["Semi-Ind"]["Avg Size"][t].append(
#                 avg_set_size(pred_sets_t)
#             )
#             results[method][GNN_model]["Semi-Ind"]["Coverage"][t].append(
#                 coverage(pred_sets_t, data, test_mask_t)
#             )

#         avg_test_acc = np.mean(
#             results[method][GNN_model]["Semi-Ind"]["Accuracy"]["All"]
#         )
#         print(f"Test accuracy: {avg_test_acc:0.3f}")

# # %% [markdown]
# # Save results to pickle file.

# # %%
# with open(results_file, "wb") as file:
#     pickle.dump(results, file)


# # # Save results
# # def save_results(params, metrics, file_path):
# #     results = {"parameters": params, "metrics": metrics}
# #     with open(file_path, "wb") as f:
# #         pickle.dump(results, f)


# # # Run and save experiment
# # metrics = run_experiment(EXPERIMENT_PARAMS)
# # save_results(EXPERIMENT_PARAMS, metrics, results_file)
