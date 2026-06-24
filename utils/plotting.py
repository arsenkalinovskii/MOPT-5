import matplotlib.pyplot as plt
import numpy as np


def plot_dataset(X, y, filename, y_true=None, title=None):
    plt.figure(figsize=(8, 5))

    plt.scatter(X, y, label="Noisy data")

    if y_true is not None:
        plt.plot(X, y_true, label="True function")

    if title:
        plt.title(title)

    plt.legend()
    plt.grid()

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_regression(model, X, y, filename, y_true=None, title=None):
    plt.figure(figsize=(8, 5))

    plt.scatter(X, y, label="Data")

    if y_true is not None:
        plt.plot(X, y_true, label="True function")

    X_plot = np.linspace(X.min(), X.max(), 1000)
    y_plot = model.predict(X_plot)

    plt.plot(X_plot, y_plot, label="Prediction")

    if title:
        plt.title(title)

    plt.legend()
    plt.grid()

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_loss(history, filename, title=None):
    plt.figure(figsize=(8, 5))

    plt.plot(history["loss"], label="Loss")

    if title:
        plt.title(title)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid()

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_loss_components(history, filename, title=None):
    plt.figure(figsize=(8, 5))

    plt.plot(history["loss"], label="Loss")
    plt.plot(history["risk"], label="Risk")
    plt.plot(history["l1"], label="L1")
    plt.plot(history["l2"], label="L2")

    if title:
        plt.title(title)

    plt.xlabel("Iteration")
    plt.ylabel("Value")

    plt.legend()
    plt.grid()

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_methods(histories, filename, metric="loss", title=None):
    plt.figure(figsize=(8, 5))

    for name, history in histories.items():
        plt.plot(history[metric], label=name)

    if title:
        plt.title(title)

    plt.xlabel("Iteration")
    plt.ylabel(metric)

    plt.legend()
    plt.grid()

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_batch_experiment(results, filename, title=None):
    plt.figure(figsize=(8, 5))

    for batch_size, history in results.items():
        x = np.arange(len(history["loss"]))
        plt.plot(x, history["loss"], label=f"batch={batch_size}")

    if title:
        plt.title(title)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid()

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_regularization_comparison(models, X, y, true_function, filename, title):
    plt.figure(figsize=(10, 6))
    x_plot = np.linspace(X.min(), X.max(), 400)
    plt.scatter(X, y, label="Noisy data", s=20)
    plt.plot(x_plot, true_function(x_plot), label="True function", linewidth=3)

    for name, model in models.items():
        Phi = model.design_matrix(x_plot)
        prediction = Phi @ model.weights

        plt.plot(x_plot, prediction, label=name, linewidth=2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
