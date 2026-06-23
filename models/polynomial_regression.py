import numpy as np

from .base_model import BaseModel


class PolynomialRegression(BaseModel):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        self.scale = None

    def fit_scaler(self, X):
        self.scale = np.max(np.abs(X))

    def n_features(self):
        return self.degree + 1

    def design_matrix(self, x):
        self.fit_scaler(x)
        scaled_x = x / self.scale
        return np.column_stack([scaled_x ** i for i in range(self.degree + 1)])
