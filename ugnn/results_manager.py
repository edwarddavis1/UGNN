import os
from datetime import datetime
import pickle


class ResultsManager:
    def __init__(self, experiment_name, base_dir="results"):
        """
        Initialize the ResultsManager.

        Args:
            experiment_name (str): Name of the experiment.
            base_dir (str): Base directory for saving results.
        """
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.results_dir = f"{base_dir}/{experiment_name}"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = f"{self.results_dir}/experiment_{self.timestamp}.pkl"

        # Create the results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

    def save_results(self, results):
        """
        Save the results to a file.

        Args:
            results (dict): The results to save.
        """
        with open(self.results_file, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {self.results_file}")

    def get_results_file_path(self):
        """
        Get the path to the results file.

        Returns:
            str: Path to the results file.
        """
        return self.results_file
