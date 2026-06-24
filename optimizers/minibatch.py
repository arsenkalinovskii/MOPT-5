import time

import numpy as np

from .base_optimizer import BaseOptimizer


class MiniBatchGD(BaseOptimizer):
    def __init__(self, lr=0.01, batch_size=32, epochs=500, tol=1e-8, random_state=67):
        super().__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.tol = tol
        self.random_state = random_state

    def fit(self, model, loss_fn, Phi, y):
        self.reset_history()

        self.history["grad_calls"] = []
        self.history["step_loss"] = []
        self.history["step_time"] = []

        rng = np.random.default_rng(self.random_state)
        start_time = time.perf_counter()
        n = len(Phi)

        grad_calls = 0

        for epoch in range(self.epochs):
            indices = rng.permutation(n)

            for batch_start in range(0, n, self.batch_size):
                batch_idx = indices[batch_start: batch_start + self.batch_size]

                Xb = Phi[batch_idx]
                yb = y[batch_idx]

                grad = loss_fn.gradient(model.weights, Xb, yb)
                model.weights -= self.lr * grad

                grad_calls += 1

                elapsed = time.perf_counter() - start_time

                loss = loss_fn.evaluate(model.weights, Phi, y)["loss"]

                self.history["step_loss"].append(loss)
                self.history["grad_calls"].append(grad_calls)
                self.history["step_time"].append(elapsed)

            elapsed = time.perf_counter() - start_time
            self.log_step(model, loss_fn, Phi, y, elapsed)

            if np.linalg.norm(grad) < self.tol:
                break

        return model, self.history
