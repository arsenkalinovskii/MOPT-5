import os
import time

import numpy as np
import pandas as pd

from optimizers import *
from losses import *
from regularization import *
from models import *

from utils.metrics import evaluate_regression
from utils.plotting import *


def linear_like_function(x):
    return 2 * x + 1 + 0.1 * np.sin(5 * x)


def nonlinear_function(x):
    return np.sin(2 * x) + 0.2 * x ** 3 + np.exp(-(x - 1) ** 2)


def generate_dataset(func, x_min=-3, x_max=3, n=200, noise_std=0.8, seed=67):
    rng = np.random.default_rng(seed)

    X = np.linspace(x_min, x_max, n)
    y_true = func(X)
    y = y_true + rng.normal(0, noise_std, n)

    return X, y, y_true


def task1():
    result_dir = 'task1'
    os.makedirs(result_dir, exist_ok=True)

    datasets = {
        "linear": linear_like_function,
        "nonlinear": nonlinear_function
    }

    results = []

    for dataset_name, func in datasets.items():
        X, y, y_true = generate_dataset(func)

        plot_dataset(
            X,
            y,
            os.path.join(result_dir, f"dataset_{dataset_name}.png"),
            y_true=y_true,
            title=f"{dataset_name} dataset"
        )

        models = [
            ("linear_regression", LinearRegression()),
            ("poly2", PolynomialRegression(2)),
            ("poly3", PolynomialRegression(3)),
            ("poly4", PolynomialRegression(4)),
            ("poly5", PolynomialRegression(5))
        ]

        for model_name, model in models:
            model.initialize_weights()
            Phi = model.design_matrix(X)

            optimizer_factories = {
                "Analytic": lambda: AnalyticSolver(),
                "SGD": lambda: SGD(lr=0.001, epochs=1000),
                "MiniBatch": lambda: MiniBatchGD(lr=0.001, batch_size=16, epochs=500),
                "GaussNewton": lambda: GaussNewton(max_iter=100),
                "LevenbergMarquardt": lambda: LevenbergMarquardt(max_iter=100)
            }

            best_loss = np.inf
            best_model = None

            for optimizer_name, optimizer_factory in optimizer_factories.items():
                optimizer = optimizer_factory()

                current_model = LinearRegression() if model_name == "linear_regression" else PolynomialRegression(
                    model.degree)

                current_model.initialize_weights()
                loss_fn = MSELoss()
                start_time = time.perf_counter()
                current_model, history = optimizer.fit(current_model, loss_fn, Phi, y)

                if model_name == "poly5" and optimizer_name != "Analytic":
                    plot_loss(
                        history,
                        os.path.join(
                            result_dir,
                            f"loss_{optimizer_name.lower()}_{dataset_name}.png"
                        ),
                        title=f"{optimizer_name} ({dataset_name})"
                    )

                if model_name == "poly5":
                    plot_regression(
                        current_model,
                        X,
                        y,
                        os.path.join(
                            result_dir,
                            f"poly5_{dataset_name}_{optimizer_name.lower()}.png"
                        ),
                        y_true=y_true,
                        title=f"Poly5 + {optimizer_name} ({dataset_name})"
                    )

                elapsed = time.perf_counter() - start_time
                metrics = evaluate_regression(current_model, X, y)
                final_loss = history["loss"][-1]

                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "optimizer": optimizer_name,
                    "loss": final_loss,
                    "iterations": len(history["loss"]),
                    "time": elapsed,
                    "r2": metrics["R2"]
                })

                if final_loss < best_loss:
                    best_loss = final_loss
                    best_model = current_model

            plot_regression(
                best_model,
                X,
                y,
                os.path.join(result_dir, f"{model_name}_{dataset_name}_function.png"),
                y_true=y_true,
                title=f"{model_name} ({dataset_name})"
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(result_dir, "results.csv"), index=False)

    for dataset_name in ["linear", "nonlinear"]:
        dataset_df = results_df[results_df["dataset"] == dataset_name].copy()
        labels = dataset_df["model"] + "\n" + dataset_df["optimizer"]

        plt.figure(figsize=(14, 6))
        plt.bar(np.arange(len(dataset_df)), dataset_df["loss"])
        plt.xticks(np.arange(len(dataset_df)), labels, rotation=90)
        plt.ylabel("Loss")
        plt.title(f"Loss comparison ({dataset_name})")

        min_val = dataset_df["loss"].min()
        max_val = dataset_df["loss"].max()
        margin = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
        plt.ylim(min_val - margin, max_val + margin)

        plt.tight_layout()
        plt.savefig(
            os.path.join(result_dir, f"loss_comparison_{dataset_name}.png"),
            bbox_inches="tight"
        )
        plt.close()

        plt.figure(figsize=(14, 6))
        plt.bar(np.arange(len(dataset_df)), dataset_df["r2"])
        plt.xticks(np.arange(len(dataset_df)), labels, rotation=90)
        plt.ylabel("R2")
        plt.title(f"R2 comparison ({dataset_name})")

        min_val = dataset_df["r2"].min()
        max_val = dataset_df["r2"].max()
        margin = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
        plt.ylim(min_val - margin, max_val + margin)

        plt.tight_layout()
        plt.savefig(
            os.path.join(result_dir, f"r2_comparison_{dataset_name}.png"),
            bbox_inches="tight"
        )
        plt.close()

    optimizer_summary = (
        results_df
        .groupby("optimizer")
        .agg({
            "loss": "mean",
            "iterations": "mean",
            "time": "mean"
        })
        .round(4)
    )

    optimizer_summary.to_csv(
        os.path.join(result_dir, "optimizer_summary.csv")
    )

    summary_df = (
        results_df
        .groupby(["dataset", "optimizer"])
        .agg({
            "loss": "mean",
            "iterations": "mean",
            "time": "mean",
            "r2": "mean"
        })
        .reset_index()
    )

    summary_df.to_csv(
        os.path.join(result_dir, "summary.csv"),
        index=False
    )

    print("Task 1 completed.")


if __name__ == "__main__":
    task1()
