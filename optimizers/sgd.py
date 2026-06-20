import time
import numpy as np

from base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):
    def __init__(self, lr=0.01, epochs=500, tol=1e-8, random_state=42):
        super().__init__()

        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.random_state = random_state

    def fit(self, model, loss_fn, X, y):
        self.reset_history()

        rng = np.random.default_rng(self.random_state)
        start_time = time.perf_counter()

        for epoch in range(self.epochs):
            idx = rng.integers(0, len(X))

            Xi = X[idx: idx + 1]
            yi = y[idx: idx + 1]

            grad = loss_fn.gradient(model.weights, Xi, yi)

            model.weights -= self.lr * grad
            elapsed = time.perf_counter() - start_time

            self.log_step(model, loss_fn, X, y, elapsed, )

            if np.linalg.norm(grad) < self.tol:
                break

        return model, self.history
