# Experiment management and training loops
class Experiment:
    def __init__(self, dataset, model, optimizer, masks, params):
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.masks = masks
        self.params = params
        self.results = {}

    def train(self):
        max_valid_acc = 0
        best_model = None
        for epoch in range(self.params["num_epochs"]):
            train_loss = train(
                self.model, self.dataset, self.masks["train"], self.optimizer
            )
            valid_acc = valid(self.model, self.dataset, self.masks["valid"])
            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                best_model = copy.deepcopy(self.model)
        return best_model, max_valid_acc

    def evaluate(self, best_model):
        output = best_model(self.dataset.x, self.dataset.edge_index)
        # Add evaluation logic here (e.g., accuracy, coverage, etc.)
        return output


####################
## Better idea for experiment management [below]
####################


# import os
# import pickle
# from datetime import datetime
# from ugnn.config import EXPERIMENT_PARAMS

# # Create results directory
# experiment_name = "SBM_with_less_test"
# results_dir = f"results/{experiment_name}"
# os.makedirs(results_dir, exist_ok=True)

# # Generate a unique file name
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# results_file = f"{results_dir}/experiment_{timestamp}.pkl"


# # Run experiment (placeholder for actual logic)
# def run_experiment(params):
#     # Example experiment logic
#     metrics = {"accuracy": 0.95, "loss": 0.1}  # Replace with actual metrics
#     return metrics


# # Save results
# def save_results(params, metrics, file_path):
#     results = {"parameters": params, "metrics": metrics}
#     with open(file_path, "wb") as f:
#         pickle.dump(results, f)


# # Run and save experiment
# metrics = run_experiment(EXPERIMENT_PARAMS)
# save_results(EXPERIMENT_PARAMS, metrics, results_file)

# print(f"Results saved to {results_file}")
