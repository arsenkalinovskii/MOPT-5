import numpy as np


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - ss_res / ss_tot


def evaluate_regression(model, X, y):
    y_pred = model.predict(X)

    return {
        "R2": r2_score(y, y_pred)
    }
