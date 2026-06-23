import time

import numpy as np

from .base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):
    def __init__(self, lr=0.001, epochs=500, tol=1e-8, random_state=42):
        super().__init__()

        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.random_state = random_state

    def fit(self, model, loss_fn, Phi, y):
        self.reset_history()

        rng = np.random.default_rng(self.random_state)
        start_time = time.perf_counter()

        for epoch in range(self.epochs):
            indices = rng.permutation(len(Phi))

            for idx in indices:
                Xi = Phi[idx:idx + 1]
                yi = y[idx:idx + 1]

                grad = loss_fn.gradient(model.weights, Xi, yi)

                model.weights -= self.lr * grad

            elapsed = time.perf_counter() - start_time
            self.log_step(model, loss_fn, Phi, y, elapsed)

        return model, self.history
