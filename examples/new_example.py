# %%
import pyemb as eb
from itertools import product
import numpy as np

from ugnn.config import EXPERIMENT_PARAMS
from ugnn.utils.masks import mask_split, non_zero_degree_mask
from ugnn.networks import Dynamic_Network, Block_Diagonal_Network, Unfolded_Network
from ugnn.experiments import Experiment
from ugnn.results_manager import ResultsManager

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

# Initialize ResultsManager
experiment_name = "example_experiment"
results_manager = ResultsManager(experiment_name)

# %%
# Generate dataset
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

# Convert the data from adjacency matrices and labels to torch geometric datasets
dataset = Dynamic_Network(As, node_labels)
dataset_BD = Block_Diagonal_Network(dataset)[0]
dataset_UA = Unfolded_Network(dataset)[0]

# Remove nodes with zero degree for each time point
data_mask = non_zero_degree_mask(As, n, T)

# %%
# Define methods, models, and regimes
methods = ["BD", "UA"]
GNN_models = ["GCN", "GAT"]
regimes = ["Assisted Semi-Ind", "Trans", "Semi-Ind"]

# Run experiments
for regime in regimes:
    num_train = num_train_trans if regime != "Semi-Ind" else num_train_semi_ind
    mode = regime.lower().replace(" ", "-")

    for method, GNN_model in product(methods, GNN_models):
        for i in range(num_train):
            # Split data into training/validation/calibration/test
            train_mask, valid_mask, calib_mask, test_mask = mask_split(
                data_mask, props, seed=i, mode=mode
            )

            # Select dataset based on method
            if method == "BD":
                data = dataset_BD
            elif method == "UA":
                data = dataset_UA
                # Pad masks to include anchor nodes
                train_mask = np.concatenate((np.array([False] * n), train_mask))
                valid_mask = np.concatenate((np.array([False] * n), valid_mask))
                calib_mask = np.concatenate((np.array([False] * n), calib_mask))
                test_mask = np.concatenate((np.array([False] * n), test_mask))

            # Initialize experiment parameters
            params = {
                "num_epochs": num_epochs,
                "num_permute_trans": num_permute_trans,
                "alpha": alpha,
                "T": T,
                "n": n,
                "num_channels_GCN": num_channels_GCN,
                "num_channels_GAT": num_channels_GAT,
                "num_classes": num_classes,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            }

            # Initialize and run the experiment
            experiment = Experiment(
                method=method,
                GNN_model=GNN_model,
                mode=regime,
                data=data,
                masks={
                    "train": train_mask,
                    "valid": valid_mask,
                    "calib": calib_mask,
                    "test": test_mask,
                },
                params=params,
            )

            experiment.train()
            experiment.evaluate()

            # Save results for this experiment
            results_manager.save_results(experiment.results)

# %%
# Save final results
print(f"All results saved to: {results_manager.get_results_file_path()}")
