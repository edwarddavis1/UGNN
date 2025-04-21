import os
import pickle
import pandas as pd


data_selection = "sbm"

# Get the most recent results file for the experiment
results_dir = f"results/{data_selection}"
results_files = [f for f in os.listdir(results_dir) if f.endswith(".pkl")]
results_files.sort(reverse=False)
results_file = os.path.join(results_dir, results_files[0])

with open(results_file, "rb") as f:
    data = pickle.load(f)

all_time_results = pd.DataFrame(data)

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


aggregated_results.sort_values(by=["Regime", "GNN Model", "Method"], inplace=True)
aggregated_results
