import os
from datetime import datetime
import pickle
from typing import Type

from ugnn.config import ROOT_DIR
from ugnn.experiments import Experiment

# TODO: Make sure to save the config with the results


class ResultsManager:
    def __init__(self, experiment_name: str):
        """
        Initialize the ResultsManager.

        Args:
            experiment_name (str): Name of the experiment.
            base_dir (str): Base directory for saving results. Defaults to 'results/' in the project root.
        """
        # Ensure results are saved in UGNN/results
        self.base_dir = os.path.join(ROOT_DIR, "../results")

        self.all = []
        self.experiment_name = experiment_name
        self.results_dir = f"{self.base_dir}/{experiment_name}"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = f"{self.results_dir}/experiment_{self.timestamp}.pkl"
        self.experiment_params = None

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

        if self.experiment_params is None:
            self.experiment_params = exp.params

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
        with open(f"{self.results_dir}/params_{self.timestamp}.pkl", "wb") as f:
            pickle.dump(self.experiment_params, f)

        print(f"Results saved to {self.results_file}")

    def return_df(self):
        """
        Return the results as a DataFrame.

        Returns:
            pd.DataFrame: The results DataFrame.
        """
        import pandas as pd

        return pd.DataFrame(self.all)
