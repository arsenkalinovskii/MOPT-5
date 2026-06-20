from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    def __init__(self):
        self.weights = None

    @abstractmethod
    def design_matrix(self, X):
        pass

    def initialize_weights(self):
        self.weights = np.zeros(self.n_features())

    def predict(self, X):
        return self.design_matrix(X) @ self.weights

    def jacobian(self, X):
        return self.design_matrix(X)

    @abstractmethod
    def n_features(self):
        pass
