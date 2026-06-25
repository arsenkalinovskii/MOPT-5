import os

import matplotlib.pyplot as plt
import numpy as np


def plot_optimizer_losses(history, optimizer_name, dataset_name, result_dir):
    plot_loss(
        history,
        os.path.join(result_dir, f"loss_{optimizer_name.lower()}_{dataset_name}.png"),
        title=f"{optimizer_name} ({dataset_name})"
    )


def plot_poly5_regression(model, X, y, optimizer_name, dataset_name, result_dir, y_true=None):
    plot_regression(
        model,
        X,
        y,
        os.path.join(result_dir, f"poly5_{dataset_name}_{optimizer_name.lower()}.png"),
        y_true=y_true,
        title=f"Poly5 + {optimizer_name} ({dataset_name})"
    )


def plot_best_model(model, X, y, model_name, dataset_name, result_dir, y_true=None):
    plot_regression(
        model,
        X,
        y,
        os.path.join(result_dir, f"{model_name}_{dataset_name}_function.png"),
        y_true=y_true,
        title=f"{model_name} ({dataset_name})"
    )


def plot_metric_comparison(dataset_df, metric, dataset_name, result_dir):
    labels = dataset_df["model"] + "\n" + dataset_df["optimizer"]

    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(dataset_df)), dataset_df[metric])

    plt.xticks(np.arange(len(dataset_df)), labels, rotation=90)
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} comparison ({dataset_name})")

    min_val = dataset_df[metric].min()
    max_val = dataset_df[metric].max()

    margin = (max_val - min_val) * 0.1 if max_val > min_val else 0.1

    plt.ylim(min_val - margin, max_val + margin)

    plt.tight_layout()

    plt.savefig(
        os.path.join(result_dir, f"{metric}_comparison_{dataset_name}.png"),
        bbox_inches="tight"
    )

    plt.close()


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
