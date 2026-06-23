import numpy as np


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - ss_res / ss_tot


def evaluate_regression(model, X, y):
    y_pred = model.predict(X)

    return {
        "MSE": mse(y, y_pred),
        "RMSE": rmse(y, y_pred),
        "MAE": mae(y, y_pred),
        "R2": r2_score(y, y_pred)
    }