import numpy as np
from optimizers import *
from losses import *
from regularization import *


def generate_dataset(func, x_min, x_max, n_points=200, noise_std=0.1, seed=42):
    rng = np.random.default_rng(seed)

    X = np.linspace(x_min, x_max, n_points)

    y_true = func(X)
    noise = rng.normal(0, noise_std, n_points)

    y = y_true + noise

    return X, y, y_true


model = PolynomialRegression(degree=5)
model.initialize_weights()

loss = MSELoss(
    L2Regularization(1e-3)
)

optimizer = MiniBatchGD(
    lr=1e-2,
    batch_size=16
)

model, history = optimizer.fit(
    model,
    loss,
    model.design_matrix(X),
    y
)
