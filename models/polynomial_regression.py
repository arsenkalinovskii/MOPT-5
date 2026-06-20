import numpy as np

from base_model import BaseModel


class PolynomialRegression(BaseModel):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree

    def design_matrix(self, X):
        X = np.asarray(X).reshape(-1)
        return np.column_stack([X ** p for p in range(self.degree + 1)])

    def n_features(self):
        return self.degree + 1

    def __repr__(self):
        return f"PolynomialRegression(degree={self.degree})"
