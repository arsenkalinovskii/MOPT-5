import numpy as np

from .base_optimizer import BaseOptimizer


class AnalyticSolver(BaseOptimizer):
    def fit(self, model, loss_fn, Phi, y):
        self.reset_history()

        model.weights = np.linalg.solve(Phi.T @ Phi, Phi.T @ y)

        self.log_step(model, loss_fn, Phi, y, 0.0)

        return model, self.history
