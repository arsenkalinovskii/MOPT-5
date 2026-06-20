import numpy as np


class NoRegularization:
    def value(self, w):
        return 0.0

    def gradient(self, w):
        return np.zeros_like(w)

    def l1_term(self, w):
        return 0.0

    def l2_term(self, w):
        return 0.0


class L1Regularization:
    def __init__(self, lam):
        self.lam = lam

    def value(self, w):
        return self.lam * np.sum(np.abs(w))

    def gradient(self, w):
        return self.lam * np.sign(w)

    def l1_term(self, w):
        return self.value(w)

    def l2_term(self, w):
        return 0.0


class L2Regularization:
    def __init__(self, lam):
        self.lam = lam

    def value(self, w):
        return self.lam * np.sum(w ** 2)

    def gradient(self, w):
        return 2 * self.lam * w

    def l1_term(self, w):
        return 0.0

    def l2_term(self, w):
        return self.value(w)


class ElasticNetRegularization:
    def __init__(self, lam1, lam2):
        self.lam1 = lam1
        self.lam2 = lam2

    def value(self, w):
        return self.lam1 * np.sum(np.abs(w)) + self.lam2 * np.sum(w ** 2)

    def gradient(self, w):
        return self.lam1 * np.sign(w) + 2 * self.lam2 * w

    def l1_term(self, w):
        return self.lam1 * np.sum(np.abs(w))

    def l2_term(self, w):
        return self.lam2 * np.sum(w ** 2)
