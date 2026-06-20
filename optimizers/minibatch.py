import time
import numpy as np

from base_optimizer import BaseOptimizer


class MiniBatchGD(BaseOptimizer):
    def __init__(self, lr=0.01, batch_size=32, epochs=500, tol=1e-8, random_state=42):
        super().__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.tol = tol
        self.random_state = random_state

    def fit(self, model, loss_fn, X, y):
        self.reset_history()

        rng = np.random.default_rng(self.random_state)
        start_time = time.perf_counter()
        n = len(X)

        for epoch in range(self.epochs):
            indices = rng.permutation(n)

            for batch_start in range(0, n, self.batch_size):
                batch_idx = indices[batch_start: batch_start + self.batch_size]

                Xb = X[batch_idx]
                yb = y[batch_idx]

                grad = loss_fn.gradient(model.weights, Xb, yb, )
                model.weights -= self.lr * grad

            elapsed = time.perf_counter() - start_time
            self.log_step(model, loss_fn, X, y, elapsed, )

            if np.linalg.norm(grad) < self.tol:
                break

        return model, self.history
