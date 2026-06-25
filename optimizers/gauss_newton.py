import time
import numpy as np

from .base_optimizer import BaseOptimizer


class GaussNewton(BaseOptimizer):
    def __init__(self, max_iter=100, tol=1e-8):
        super().__init__()

        self.max_iter = max_iter
        self.tol = tol

    def fit(self, model, loss_fn, Phi, y):
        self.reset_history()

        start_time = time.perf_counter()

        for _ in range(self.max_iter):
            residuals = Phi @ model.weights - y

            A = Phi.T @ Phi
            b = -(Phi.T @ residuals)

            step = np.linalg.solve(A, b)

            current_loss = loss_fn.evaluate(model.weights, Phi, y)["loss"]
            candidate_weights = model.weights + step
            candidate_loss = loss_fn.evaluate(candidate_weights, Phi, y)["loss"]

            model.weights = candidate_weights

            elapsed = time.perf_counter() - start_time

            self.log_step(model, loss_fn, Phi, y, elapsed)

            if np.linalg.norm(step) < self.tol:
                break
            if candidate_loss >= current_loss and np.linalg.norm(step) < self.tol:
                break
            if abs(current_loss - candidate_loss) < self.tol:
                break

        return model, self.history

    def __str__(self):
        return self.__class__.__name__
