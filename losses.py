import numpy as np

from regularization import NoRegularization


class MSELoss:
    def __init__(self, regularization=None):
        self.regularization = regularization or NoRegularization()

    def risk(self, w, X, y):
        preds = X @ w
        residuals = preds - y

        return 0.5 * np.mean(residuals ** 2)

    def gradient(self, w, X, y):
        n = len(X)

        preds = X @ w
        residuals = preds - y

        grad_risk = X.T @ residuals / n
        grad_reg = self.regularization.gradient(w)

        return grad_risk + grad_reg

    def evaluate(self, w, X, y):
        risk = self.risk(w, X, y)

        l1 = self.regularization.l1_term(w)
        l2 = self.regularization.l2_term(w)

        loss = risk + l1 + l2

        return {
            "loss": loss,
            "risk": risk,
            "l1": l1,
            "l2": l2
        }
