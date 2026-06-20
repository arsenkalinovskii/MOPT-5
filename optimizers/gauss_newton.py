import time

import numpy as np

from base_optimizer import BaseOptimizer


class GaussNewton(BaseOptimizer):
    def __init__(self, max_iter=100, tol=1e-8):
        super().__init__()

        self.max_iter = max_iter
        self.tol = tol

    def fit(self, model, loss_fn, X, y):
        self.reset_history()
        start_time = time.perf_counter()

        for _ in range(self.max_iter):
            residuals = model.predict(X) - y

            J = model.jacobian(X)
            A = J.T @ J
            b = -(J.T @ residuals)

            step = np.linalg.solve(A, b)
            model.weights += step
            elapsed = time.perf_counter() - start_time

            self.log_step(model, loss_fn, X, y, elapsed)

            if np.linalg.norm(step) < self.tol:
                break

        return model, self.history
