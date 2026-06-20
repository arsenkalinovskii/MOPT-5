from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    def __init__(self):
        self.history = {
            "loss": [],
            "risk": [],
            "l1": [],
            "l2": [],
            "time": [],
        }

    def reset_history(self):
        for key in self.history:
            self.history[key] = []

    def log_step(self, model, loss_fn, X, y, elapsed):
        metrics = loss_fn.evaluate(model.weights, X, y)

        self.history["loss"].append(metrics["loss"])
        self.history["risk"].append(metrics["risk"])
        self.history["l1"].append(metrics["l1"])
        self.history["l2"].append(metrics["l2"])
        self.history["time"].append(elapsed)

    @abstractmethod
    def fit(self, model, loss_fn, X, y):
        pass
