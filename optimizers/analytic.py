import numpy as np

from base_optimizer import BaseOptimizer


class AnalyticSolver(BaseOptimizer):
    def fit(self, model, loss_fn, X, y):
        self.reset_history()

        if X.shape[1] != 2:
            raise ValueError(
                "Analytic solution implemented only for linear regression"
            )

        x = X[:, 1]

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        w1 = numerator / denominator
        w0 = y_mean - w1 * x_mean

        model.weights = np.array([w0, w1])
        self.log_step(model, loss_fn, X, y, 0.0)

        return model, self.history
