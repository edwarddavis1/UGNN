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
# %%

regime = "semi-inductive"

res = ResultsManager(
    experiment_name=experiment_name,
)

for method, GNN_model in product(methods, GNN_models):
    for i in range(num_train_semi_ind):
        # Randomly separate remaining nodes according to data split proportions and regime
        train_mask, valid_mask, calib_mask, test_mask = mask_split(
            data_mask, props, seed=i, regime=regime
        )

        if method == "BD":
            method_str = "Block Diagonal"
            data = dataset_BD
        elif method == "UA":
            method_str = "Unfolded"
            data = dataset_UA
            # Pad masks to include anchor nodes
            train_mask = np.concatenate((np.array([False] * n), train_mask))
            valid_mask = np.concatenate((np.array([False] * n), valid_mask))
            calib_mask = np.concatenate((np.array([False] * n), calib_mask))
            test_mask = np.concatenate((np.array([False] * n), test_mask))
        else:
            raise ValueError(f"Unknown method: {method}")

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

        # Add this experiment run to the results manager
        res.add_result(exp)

# Save all the results
# res.save_results()

# %%
# Quickly just print out the results for the semi-inductive regime
# So can make sure that the results look correct

import pandas as pd

assert regime == "semi-inductive"
num_vals = EXPERIMENT_PARAMS["num_train_semi_ind"]

all_time_results = pd.DataFrame(res.all)
all_time_results = all_time_results[all_time_results["Time"] == "All"].drop(
    columns=["Time"]
)

# Group by the unique triples (method, GNN_model, regime)
grouped = all_time_results.groupby(["Method", "GNN Model", "Regime"])
aggregated_results = (
    grouped[["Accuracy", "Avg Size", "Coverage"]].agg(["mean", "std"]).reset_index()
)
aggregated_results.columns = [
    "_".join(col).strip("_") for col in aggregated_results.columns
]

print("\n\n")
print(aggregated_results)
