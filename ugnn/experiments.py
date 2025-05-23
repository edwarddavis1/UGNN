import numpy as np
import torch
import copy
import argparse

from typing import Literal
from torch_geometric.data import Data
from ugnn.types import ExperimentParams, Masks, DataParams

from data import get_sbm_data, get_school_data, get_flight_data
from ugnn.experiment_config import (
    MINIMAL_EXPERIMENT_PARAMS,
    SBM_EXPERIMENT_PARAMS,
    SCHOOL_EXPERIMENT_PARAMS,
    FLIGHT_EXPERIMENT_PARAMS,
)

from ugnn.gnns import GCN, GAT, train, valid
from ugnn.utils.metrics import accuracy, avg_set_size, coverage
from ugnn.conformal import get_prediction_sets
from ugnn.utils.masks import mask_mix


class Experiment:
    def __init__(
        self,
        method: Literal["BD", "UA"],
        GNN_model: Literal["GCN", "GAT"],
        regime: Literal["transductive", "semi-inductive", "temporal transductive"],
        data: Data,
        masks: Masks,
        experiment_params: ExperimentParams,
        data_params: DataParams,
        conformal_method: Literal["APS", "RAPS", "SAPS"] = "APS",
        seed: int = 123,
    ):
        """
        Initializes the experiment with the specified parameters.

        This class trains a GNN on multiple networks (e.g., a discrete-time dynamic network)
        and performs conformal prediction with the GNN. The experiment evaluates the GNN's
        performance using accuracy, average prediction set size, and coverage.

        The data is split into train, validation, calibration, and test groups based on the
        specified regime:

        - **Transductive**: Nodes are randomly assigned to train/valid/calib/test groups.
        - **Semi-inductive**: Nodes after a certain time point are assigned to the test group,
          while earlier nodes are randomly assigned to train/valid/calib groups.
        - **Temporal transductive**: Nodes after a certain time point are split between calib
          and test groups, while earlier nodes are assigned to train/valid groups.

        Args:
            method (Literal["BD", "UA"]): The method to represent multiple networks as a single
                network ("block diagonal" or "unfolded").
            GNN_model (Literal["GCN", "GAT"]): The GNN model to use.
            regime (Literal["transductive", "semi-inductive", "temporal transductive"]): The
                experiment regime.
            data (Data): The dataset object containing graph data and labels.
            masks (Masks): A dictionary with train, validation, calibration, and test masks.
            experiment_params (ExperimentParams): Parameters for the experiment (e.g., number
                of epochs, learning rate, etc.).
            data_params (DataParams): Parameters for the dataset (e.g., number of nodes, time
                steps, and classes).
        """
        self.method = method
        self.GNN_model = GNN_model
        self.regime = regime
        self.data = data
        self.masks = masks
        self.params = experiment_params
        self.conformal_method = conformal_method
        self.seed = seed

        # Data params
        self.n = data_params["n"]
        self.T = data_params["T"]
        self.num_classes = data_params["num_classes"]

        self.results = {
            "Accuracy": {"All": [], "Per Time": {t: [] for t in range(self.T)}},
            "Avg Size": {"All": [], "Per Time": {t: [] for t in range(self.T)}},
            "Coverage": {"All": [], "Per Time": {t: [] for t in range(self.T)}},
        }

    def initialise_model(self):
        """
        Initialise the GNN model based on the specified type.
        """
        if self.GNN_model == "GCN":
            return GCN(
                self.data.num_nodes,
                self.params["num_channels_GCN"],
                self.num_classes,
                seed=self.seed,
            )
        elif self.GNN_model == "GAT":
            return GAT(
                self.data.num_nodes,
                self.params["num_channels_GAT"],
                self.num_classes,
                seed=self.seed,
            )

    def train(self):
        """
        Train the GNN model.
        """
        model = self.initialise_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )

        # print(f"\nTraining {self.method} {self.GNN_model} in {self.regime} regime")
        max_valid_acc = 0
        for epoch in range(self.params["num_epochs"]):
            _ = train(model, self.data, self.masks["train"], optimizer)
            valid_acc = valid(model, self.data, self.masks["valid"])

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                self.best_model = copy.deepcopy(model)

        # print(f"Validation accuracy: {max_valid_acc:0.3f}")

    def evaluate(self):
        """
        Evaluate the trained model and compute metrics.
        """
        # print(f"Evaluating {self.method} {self.GNN_model} in {self.regime} regime")
        output = self.best_model(
            self.data.x, self.data.edge_index, self.data.edge_weight
        )

        if self.regime != "semi-inductive":
            for j in range(self.params["num_permute_trans"]):
                calib_mask, test_mask = mask_mix(
                    self.masks["calib"], self.masks["test"], seed=j
                )

                pred_sets = get_prediction_sets(
                    output,
                    self.data,
                    calib_mask,
                    test_mask,
                    score_function=self.conformal_method,
                    alpha=self.params["alpha"],
                )

                self._update_results(output, pred_sets, test_mask)
        else:
            pred_sets = get_prediction_sets(
                output,
                self.data,
                self.masks["calib"],
                self.masks["test"],
                score_function=self.conformal_method,
                alpha=self.params["alpha"],
            )
            self._update_results(output, pred_sets, self.masks["test"])

    def _update_results(self, output, pred_sets, test_mask):
        """
        Update the results dictionary with metrics.

        Args:
            output: Model output.
            pred_sets: Prediction sets.
            test_mask: Test mask.
        """
        self.results["Accuracy"]["All"].append(accuracy(output, self.data, test_mask))
        self.results["Avg Size"]["All"].append(avg_set_size(pred_sets))
        self.results["Coverage"]["All"].append(
            coverage(pred_sets, self.data, test_mask)
        )
        # print("Accuracy: ", self.results["Accuracy"]["All"][-1])
        # print("Avg Size: ", self.results["Avg Size"]["All"][-1])
        # print("Coverage: ", self.results["Coverage"]["All"][-1])
        # print("-------------------------------------------------------")

        for t in range(self.T):
            test_mask_t = self._get_time_mask(test_mask, t)
            if np.sum(test_mask_t) == 0:
                continue

            pred_sets_t = self._get_time_prediction_sets(
                pred_sets, test_mask, test_mask_t
            )

            self.results["Accuracy"]["Per Time"][t].append(
                accuracy(output, self.data, test_mask_t)
            )
            self.results["Avg Size"]["Per Time"][t].append(avg_set_size(pred_sets_t))
            self.results["Coverage"]["Per Time"][t].append(
                coverage(pred_sets_t, self.data, test_mask_t)
            )

    def _get_time_mask(self, test_mask, t):
        """
        Generate a time-specific mask for test nodes.

        Args:
            test_mask (np.ndarray): The test mask.
            t (int): The time step.

        Returns:
            np.ndarray: Time-specific test mask.
        """
        if self.method == "BD":
            time_mask = np.array([[False] * self.n for _ in range(self.T)])
            time_mask[t] = True
            time_mask = time_mask.reshape(-1)
        elif self.method == "UA":
            time_mask = np.array([[False] * self.n for _ in range(self.T + 1)])
            time_mask[t + 1] = True
            time_mask = time_mask.reshape(-1)
        return time_mask * test_mask

    def _get_time_prediction_sets(self, pred_sets, test_mask, test_mask_t):
        """
        Get prediction sets corresponding to a specific time step.

        Args:
            pred_sets (np.ndarray): Prediction sets for all test nodes.
            test_mask (np.ndarray): The test mask.
            test_mask_t (np.ndarray): Time-specific test mask.

        Returns:
            np.ndarray: Prediction sets for the specific time step.
        """
        return pred_sets[
            np.array(
                [
                    np.where(np.where(test_mask)[0] == np.where(test_mask_t)[0][i])[0][
                        0
                    ]
                    for i in range(sum(test_mask_t))
                ]
            )
        ]


def parse_args_load_data():
    """
    Organise experiment parameters and data parameters for running experiments.

    Parameters for GNN fitting are defined in ugnn.experiment_config.py.
    Arguments can be modified in the command line using argparse.

    Selected data is then loaded, with matching GNN fitting params for the data.

    Returns:
        EXPERIMENT_PARAMS (ExperimentParams): Experiment parameters.
        DATA_PARAMS (DataParams): Data parameters.

    """
    parser = argparse.ArgumentParser(description="Run conformal experiment.")
    parser.add_argument(
        "--data",
        type=str,
        choices=["test", "sbm", "school", "flight"],
        default="school",
        help="Name of the experiment to run (sbm, school, or flight).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the experiment as quick as possible (for debugging).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name of the experiment run.",
    )
    parser.add_argument(
        "--regime",
        type=str,
        choices=["all", "semi-inductive", "transductive", "temporal transductive"],
        default="all",
        help="Regime of the experiment to run (semi-inductive, transductive, or temporal transductive).",
    )
    parser.add_argument(
        "--conformal_method",
        type=str,
        choices=["APS", "RAPS", "SAPS", "THR"],
        default="APS",
        help="Conformal method to use (APS, RAPS, SAPS or THR).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["all", "BD", "UA"],
        default="all",
        help="Method to use to represent the dynamic network. Either unfolded [UA] or block diagonal [BD].",
    )
    parser.add_argument(
        "--GNN",
        type=str,
        choices=["all", "GCN", "GAT"],
        default="all",
        help="GNN model to use (GCN or GAT).",
    )

    args = parser.parse_args()
    data_selection = args.data
    debug_mode = args.debug
    experiment_name = args.name if args.name != "" else f"{data_selection}_exp"

    # Load selected data
    if data_selection == "sbm":
        As, node_labels = get_sbm_data()
        EXPERIMENT_PARAMS = SBM_EXPERIMENT_PARAMS
    elif data_selection == "school":
        As, node_labels = get_school_data()
        EXPERIMENT_PARAMS = SCHOOL_EXPERIMENT_PARAMS
    elif data_selection == "flight":
        As, node_labels = get_flight_data()
        EXPERIMENT_PARAMS = FLIGHT_EXPERIMENT_PARAMS
    else:
        raise ValueError(f"Unknown data: {data_selection}")

    print(f"Loaded {data_selection} data ")

    # If in debug mode, reduce the number of epochs and training samples
    if debug_mode:
        EXPERIMENT_PARAMS = MINIMAL_EXPERIMENT_PARAMS
        EXPERIMENT_PARAMS["data"] = data_selection
        experiment_name = f"{experiment_name}_debug"

    # Data parameters
    T = As.shape[0]
    n = As[0].shape[0]
    num_classes = len(np.unique(node_labels))

    DATA_PARAMS: DataParams = {
        "As": As,
        "node_labels": node_labels,
        "n": n,
        "T": T,
        "num_classes": num_classes,
    }

    # Add in any changes to the EXPERIMENT_PARAMS from argparser
    EXPERIMENT_PARAMS["experiment_name"] = experiment_name
    EXPERIMENT_PARAMS["regimes"] = (
        [args.regime]
        if args.regime != "all"
        else ["semi-inductive", "transductive", "temporal transductive"]
    )
    EXPERIMENT_PARAMS["conformal_method"] = args.conformal_method
    EXPERIMENT_PARAMS["methods"] = (
        [args.method] if args.method != "all" else ["BD", "UA"]
    )
    EXPERIMENT_PARAMS["GNN_models"] = (
        [args.GNN] if args.GNN != "all" else ["GCN", "GAT"]
    )

    return EXPERIMENT_PARAMS, DATA_PARAMS
