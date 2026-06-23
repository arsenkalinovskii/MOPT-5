import time
import numpy as np

from .base_optimizer import BaseOptimizer


class LevenbergMarquardt(BaseOptimizer):
    def __init__(self, max_iter=100, lambda0=1e-2, lambda_inc=10.0, lambda_dec=0.3, tol=1e-8):
        super().__init__()

        self.max_iter = max_iter
        self.lambda0 = lambda0
        self.lambda_inc = lambda_inc
        self.lambda_dec = lambda_dec
        self.tol = tol

    def fit(self, model, loss_fn, Phi, y):
        self.reset_history()

        lam = self.lambda0
        start_time = time.perf_counter()

        for _ in range(self.max_iter):
            residuals = Phi @ model.weights - y

            A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
            b = -(Phi.T @ residuals)

            step = np.linalg.solve(A, b)

            current_loss = loss_fn.evaluate(model.weights, Phi, y)["loss"]
            candidate_weights = model.weights + step
            candidate_loss = loss_fn.evaluate(candidate_weights, Phi, y)["loss"]

            if candidate_loss < current_loss:
                model.weights = candidate_weights
                lam *= self.lambda_dec
            else:
                lam *= self.lambda_inc

            elapsed = time.perf_counter() - start_time
            self.log_step(model, loss_fn, Phi, y, elapsed)

            if np.linalg.norm(step) < self.tol:
                break

        return model, self.history
