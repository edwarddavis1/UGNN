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
