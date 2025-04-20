import os
from datetime import datetime
import pickle
import json
from typing import Type

from ugnn.experiment_config import ROOT_DIR
from ugnn.types import ExperimentParams
from ugnn.experiments import Experiment


class ResultsManager:
    def __init__(self, params: ExperimentParams, experiment_name: str = ""):
        """
        Initialize the ResultsManager.

        This class is responsible for managing the results of the experiments. For an experiment
        of given parameters, each run can be added to the ResultsManager object. This formats
        the results and can save them to a file or return them as a DataFrame.

        Results are saved in the `UGNN/results` directory, and placed in a subdirectory matching
        the name of the data used in training. The `experiment_name` can be used to add a
        distinguishing name to the results file. If no name is provided, this name will be
        "experiment".

        Args:
            params (ExperimentParams): Parameters for the experiment.
            experiment_name (str): Name of the experiment (included in the results file name).

        Attributes:
            base_dir (str): Base directory for saving results.
            all (list): List to store all experiment results.
            experiment_params (ExperimentParams): Parameters for the experiment.
            experiment_name (str): Name of the experiment.
            results_dir (str): Directory for saving results.
            timestamp (str): Timestamp for the results file name.
            results_file (str): Path to the results file.

        Example:
            >>> from ugnn.results_manager import ResultsManager
            >>> from ugnn.experiment_config import SCHOOL_EXPERIMENT_PARAMS
            >>> results_manager = ResultsManager(SCHOOL_EXPERIMENT_PARAMS)
            >>> results_manager.add_result(exp)
            >>> results_manager.save_results()
            >>> df = results_manager.return_df()

            Files will then be saved as
            `UGNN/results/<exp.data>/<experiment_name>_<timestamp>.pkl` and
            `UGNN/results/<exp.data>/params_<experiment_name>_<timestamp>.pkl`.
        """
        # Ensure results are saved in UGNN/results
        self.base_dir = os.path.join(ROOT_DIR, "../results")
        self.all = []
        self.experiment_params = params
        self.experiment_name = (
            experiment_name if experiment_name != "" else "experiment"
        )
        self.results_dir = f"{self.base_dir}/{params["data"]}"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = (
            f"{self.results_dir}/{self.experiment_name}_{self.timestamp}.pkl"
        )

        # Create the results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

    def _format_exp_results(self, exp: Type[Experiment]):
        """
        Experiment results give many values for each metric.
        A metric is given across [all] time windows, and also [per time] window.

        The aim of this function is to separete the single line of experiment
        results into multiple lines, one for each time window, and one for all.

        Args:
            exp (Experiment): The Experiment object containing results and metadata.
        Returns:
            list: A list of dictionaries, each containing the results for a specific time window.
        """

        data = exp.results

        rows = []
        for time_key, time_value in [("All", data["Accuracy"]["All"])] + list(
            data["Accuracy"]["Per Time"].items()
        ):
            row = {
                "Time": str(time_key),
                "Accuracy": time_value[0] if time_value else None,
                "Avg Size": (
                    data["Avg Size"]["All"][0]
                    if time_key == "All"
                    else (
                        data["Avg Size"]["Per Time"][time_key][0]
                        if data["Avg Size"]["Per Time"][time_key]
                        else None
                    )
                ),
                "Coverage": (
                    data["Coverage"]["All"][0]
                    if time_key == "All"
                    else (
                        data["Coverage"]["Per Time"][time_key][0]
                        if data["Coverage"]["Per Time"][time_key]
                        else None
                    )
                ),
            }
            # Only append the row if at least one metric is not None
            if any(
                row[metric] is not None
                for metric in ["Accuracy", "Avg Size", "Coverage"]
            ):
                rows.append(row)

        return rows

    def add_result(self, exp: Type[Experiment]):
        """
        Add the results of an experiment to the results list.

        Args:
            exp (Experiment): The Experiment object containing results and metadata.
        """

        exp_rows = self._format_exp_results(exp)

        self.all.extend(
            [
                {
                    "Method": exp.method,
                    "GNN Model": exp.GNN_model,
                    "Regime": exp.regime,
                    "Accuracy": row["Accuracy"],
                    "Avg Size": row["Avg Size"],
                    "Coverage": row["Coverage"],
                    "Time": row["Time"],
                }
                for row in exp_rows
            ]
        )

    def save_results(self):
        """
        Save the results to a file.

        Args:
            results (dict): The results to save.
        """
        with open(self.results_file, "wb") as f:
            pickle.dump(self.all, f)

        # Also save the params with the results
        with open(
            f"{self.results_dir}/params_{self.experiment_name}_{self.timestamp}.json",
            "w",
        ) as f:
            json.dump(self.experiment_params, f, indent=4)

        print(f"Results saved to {self.results_file}")

    def return_df(self):
        """
        Return the results as a DataFrame.

        Returns:
            pd.DataFrame: The results DataFrame.
        """
        import pandas as pd

        return pd.DataFrame(self.all)
