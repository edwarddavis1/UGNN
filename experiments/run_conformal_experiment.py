# %%
from itertools import product
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

from ugnn.networks import Dynamic_Network, Block_Diagonal_Network, Unfolded_Network
from ugnn.utils.masks import mask_split, non_zero_degree_mask
from ugnn.experiments import Experiment, parse_args_load_data
from ugnn.results_manager import ResultsManager

np.random.seed(42)

# %%
# Parse args and load data
EXPERIMENT_PARAMS, DATA_PARAMS = parse_args_load_data()

# Unpack parameters from EXPERIMENT_PARAMS
experiment_name = EXPERIMENT_PARAMS["experiment_name"]

# GNN and conformal params
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

# Model params
methods = EXPERIMENT_PARAMS["methods"]
GNN_models = EXPERIMENT_PARAMS["GNN_models"]
regimes = EXPERIMENT_PARAMS["regimes"]
conformal_method = EXPERIMENT_PARAMS["conformal_method"]


# Prepare results directory
data_selection = EXPERIMENT_PARAMS["data"]
results_dir = f"results/{data_selection}"
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"{results_dir}/experiment_{timestamp}.pkl"

# Unpack DATA_PARAMS
As = DATA_PARAMS["As"]
node_labels = DATA_PARAMS["node_labels"]
n = DATA_PARAMS["n"]
T = DATA_PARAMS["T"]

# %%
# Convert the data from adjacency matrices and labels to a torch geometric dataset
# This contains T separate graphs
dataset = Dynamic_Network(As, node_labels)

# Arrange the many graphs into a single graph, allowing the GNN to process at once
dataset_BD = Block_Diagonal_Network(dataset)[0]  # Block diagonal representation
dataset_UA = Unfolded_Network(dataset)[0]  # Unfolded representation


# %%
# Remove nodes with zero degree for each time point
data_mask = non_zero_degree_mask(As, n, T)

# Results manager object to keep track of each experiment run
res = ResultsManager(params=EXPERIMENT_PARAMS, experiment_name=experiment_name)

# Main experiment loop
for method, GNN_model, regime in product(methods, GNN_models, regimes):
    print(
        f"Running experiment for method: {method}, GNN model: {GNN_model}, regime: {regime}"
    )
    if regime == "semi-inductive":
        num_train = num_train_semi_ind
    elif regime == "transductive" or regime == "temporal transductive":
        num_train = num_train_trans
    else:
        raise ValueError(f"Unknown regime: {regime}")

    for i in tqdm(range(num_train)):
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
            conformal_method=conformal_method,
        )

        # Train the GNN on the data
        exp.train()

        # Compute prediction sets and evaluate performance
        exp.evaluate()

        # Add this experiment run to the results manager
        res.add_result(exp)

# Save all the results
res.save_results()

# %%

all_time_results = res.return_df()
all_time_results = all_time_results[all_time_results["Time"] == "All"].drop(
    columns=["Time"]
)
grouped = all_time_results.groupby(["Method", "GNN Model", "Regime"])
aggregated_results = (
    grouped[["Accuracy", "Avg Size", "Coverage"]].agg(["mean", "std"]).reset_index()
)
aggregated_results.columns = [
    "_".join(col).strip("_") for col in aggregated_results.columns
]

print("\n\n")
print(aggregated_results)

# %%
aggregated_results.sort_values(by=["Regime", "GNN Model", "Method"], inplace=True)
aggregated_results
