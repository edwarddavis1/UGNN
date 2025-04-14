import pickle


class ResultsManager:
    def __init__(self, methods, models, regimes, outputs, times):
        self.results = self._initialize_results(
            methods, models, regimes, outputs, times
        )

    def _initialize_results(self, methods, models, regimes, outputs, times):
        results = {}
        for method in methods:
            results[method] = {}
            for model in models:
                results[method][model] = {}
                for regime in regimes:
                    results[method][model][regime] = {}
                    for output in outputs:
                        results[method][model][regime][output] = {
                            time: [] for time in times
                        }
        return results

    def update(self, method, model, regime, output, time, value):
        self.results[method][model][regime][output][time].append(value)

    def save(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self.results, file)
