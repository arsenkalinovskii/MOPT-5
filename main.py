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


def task2():
    result_dir = "task2"
    os.makedirs(result_dir, exist_ok=True)

    X, y, y_true = generate_dataset(nonlinear_function)
    model_degree = 5
    batch_sizes = [1, 4, 8, 16, 32, 64, len(X)]

    histories = {}

    for batch_size in batch_sizes:
        model = PolynomialRegression(model_degree)
        model.initialize_weights()

        Phi = model.design_matrix(X)

        optimizer = MiniBatchGD(lr=0.03, batch_size=batch_size, epochs=100)
        loss_fn = MSELoss()

        model, history = optimizer.fit(model, loss_fn, Phi, y)
        histories[batch_size] = history

        # loss по эпохам
        plt.figure(figsize=(8, 5))
        plt.plot(history["loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Batch size = {batch_size}")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"loss_epoch_batch_{batch_size}.png"))
        plt.close()

        # loss по вычислениям градиента
        plt.figure(figsize=(8, 5))
        plt.plot(history["grad_calls"], history["step_loss"])

        plt.xlabel("Gradient evaluations")
        plt.ylabel("Loss")
        plt.title(f"Batch size = {batch_size}")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"loss_grad_batch_{batch_size}.png"))
        plt.close()

    plt.figure(figsize=(10, 6))

    for batch_size, history in histories.items():
        plt.plot(history["loss"], label=f"B={batch_size}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "loss_epoch_comparison.png"))
    plt.close()

    plt.figure(figsize=(10, 6))

    for batch_size, history in histories.items():
        line = plt.plot(history["grad_calls"], history["step_loss"], label=f"B={batch_size}")
        plt.plot(
            history["grad_calls"][-1],
            history["step_loss"][-1],
            marker='o',
            markersize=8,
            color=line[0].get_color()
        )

    plt.xlabel("Gradient evaluations")
    plt.ylabel("Loss")
    plt.title("Loss vs Gradient Evaluations")
    plt.ylim(top=1)  # Обрезаем график сверху по значению 1
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "loss_gradient_comparison.png"))
    plt.close()

    print("Task 2 completed.")


def task3():
    result_dir = "task3"
    os.makedirs(result_dir, exist_ok=True)

    X, y, y_true = generate_dataset(nonlinear_function, n=50, noise_std=1.5)

    trained_models = {}

    degree = 10
    model = PolynomialRegression(degree)
    model.initialize_weights()
    Phi = model.design_matrix(X)

    hyper_params = dict(lr=0.05, batch_size=4, epochs=1000)
    optimizer = MiniBatchGD(**hyper_params)
    loss_fn = MSELoss(regularization=NoRegularization())
    model, history = optimizer.fit(model, loss_fn, Phi, y)

    trained_models["Without regularization"] = model

    params = [0.002, 0.02, 0.2]

    configs = [
        *((idx, "L1", L1Regularization(i)) for idx, i in enumerate(params)),
        *((idx, "L2", L2Regularization(i)) for idx, i in enumerate(params)),
        *((idx, "ElasticNet", ElasticNetRegularization(i, i)) for idx, i in enumerate(params))
    ]

    for idx, reg_name, regularization in configs:
        model = PolynomialRegression(degree)
        model.initialize_weights()

        optimizer = MiniBatchGD(**hyper_params)
        loss_fn = MSELoss(regularization=regularization)
        model, history = optimizer.fit(model, loss_fn, Phi, y)

        if reg_name == "ElasticNet":
            suffix = f"l1={regularization.lam1};l2={regularization.lam2}"
        else:
            suffix = 'l=' + str(regularization.lam)

        trained_models[reg_name + '_' + suffix] = model

        plt.figure(figsize=(10, 6))
        width = 3
        plt.plot(history["loss"], label="Total loss", linewidth=width)
        plt.plot(history["risk"], label="Risk", linewidth=width)

        if reg_name == "L1":
            plt.plot(history["l1"], label="L1 penalty", linewidth=width)
        elif reg_name == "L2":
            plt.plot(history["l2"], label="L2 penalty", linewidth=width)
        else:
            plt.plot(history["l1"], label="L1 part", linewidth=width)
            plt.plot(history["l2"], label="L2 part", linewidth=width)

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(f"{reg_name}, {suffix}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"{reg_name}_{idx + 1}.png"))
        plt.close()

    without_regularization_data = {"Without regularization": trained_models["Without regularization"]}

    plot_regularization_comparison(
        without_regularization_data | {i: trained_models[i] for i in trained_models if 'L1' in i},
        X,
        y,
        nonlinear_function,
        os.path.join(result_dir, "l1_regularization.png"),
        "L1 Regularization"
    )

    plot_regularization_comparison(
        without_regularization_data | {i: trained_models[i] for i in trained_models if 'L2' in i},
        X,
        y,
        nonlinear_function,
        os.path.join(result_dir, "l2_regularization.png"),
        "L2 Regularization"
    )

    plot_regularization_comparison(
        without_regularization_data | {i: trained_models[i] for i in trained_models if 'ElasticNet' in i},
        X,
        y,
        nonlinear_function,
        os.path.join(result_dir, "elastic_net.png"),
        "Elastic Net Regularization"
    )

    coefficients = []
    abs_coefficients = []

    for name, model in trained_models.items():
        weights = model.weights

        row = {"model": name}
        abs_row = {"model": name}

        for i, w in enumerate(weights):
            row[f"w_{i}"] = w
            abs_row[f"|w_{i}|"] = abs(w)

        coefficients.append(row)
        abs_coefficients.append(abs_row)

    pd.DataFrame(coefficients).to_csv(os.path.join(result_dir, "model_coefficients.csv"), index=False)
    pd.DataFrame(abs_coefficients).to_csv(os.path.join(result_dir, "model_absolute_coefficients.csv"), index=False)

    print("Task 3 completed.")


def task4():
    result_dir = "task4"
    os.makedirs(result_dir, exist_ok=True)

    X, y, y_true = generate_dataset(nonlinear_function)

    model = PolynomialRegression(5)
    loss_fn = MSELoss()

    epochs = 100

    optimizers = {
        "SGD": lambda: SGD(lr=0.01, epochs=epochs),
        "MiniBatch": lambda: MiniBatchGD(lr=0.01, batch_size=4, epochs=epochs),
        "GaussNewton": lambda: GaussNewton(max_iter=epochs),
        "LevenbergMarquardt": lambda: LevenbergMarquardt(max_iter=epochs)
    }

    results = []
    histories = {}

    Phi = model.design_matrix(X)

    for name, optimizer_factory in optimizers.items():
        current_model = PolynomialRegression(5)
        current_model.initialize_weights()

        optimizer = optimizer_factory()
        start_time = time.perf_counter()
        current_model, history = optimizer.fit(current_model, loss_fn, Phi, y)
        elapsed = time.perf_counter() - start_time

        histories[name] = history

        results.append({
            "optimizer": str(optimizer),
            "loss": history["loss"][-1],
            "iterations": len(history["loss"]),
            "time": elapsed
        })

    pd.DataFrame(results).to_csv(os.path.join(result_dir, "optimization_comparison.csv"), index=False)

    plt.figure(figsize=(8, 5))

    for name in ["SGD", "MiniBatch"]:
        if name in histories:
            plt.plot(histories[name]["loss"], label=name)

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "loss_stochastic.png"))
    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(histories["GaussNewton"]["loss"], label="Gauss-Newton")
    plt.plot(histories["LevenbergMarquardt"]["loss"], label="Levenberg-Marquardt")

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "gn_vs_lm_loss.png"))
    plt.close()

    max_len = max(
        len(histories["GaussNewton"]["loss"]),
        len(histories["LevenbergMarquardt"]["loss"])
    )

    rows = []

    for i in range(max_len):
        rows.append({
            "iteration": i + 1,
            "gauss_newton":
                histories["GaussNewton"]["loss"][i]
                if i < len(histories["GaussNewton"]["loss"])
                else np.nan,
            "levenberg_marquardt":
                histories["LevenbergMarquardt"]["loss"][i]
                if i < len(histories["LevenbergMarquardt"]["loss"])
                else np.nan
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(result_dir, "gn_vs_lm_history.csv"),
        index=False
    )



    starts = [
        ('zeros', np.zeros(model.n_features())),
        ('ones', np.ones(model.n_features())),
        ('random', np.random.randn(model.n_features()))
    ]

    sensitivity = []

    optimizers_best = {
        "GaussNewton": lambda: GaussNewton(max_iter=100),
        "LevenbergMarquardt": lambda: LevenbergMarquardt(max_iter=100)
    }

    for start_id, w0 in starts:
        for name, optimizer_factory in optimizers_best.items():
            current_model = PolynomialRegression(5)
            current_model.weights = w0.copy()

            optimizer = optimizer_factory()
            current_model, history = optimizer.fit(current_model, loss_fn, Phi, y)

            sensitivity.append({
                "optimizer": name,
                "start_point": start_id,
                "iterations": len(history["loss"]),
                "final_loss": history["loss"][-1]
            })

    pd.DataFrame(sensitivity).to_csv(
        os.path.join(result_dir, "initialization_sensitivity.csv"),
        index=False
    )

    print("Task 4 completed.")


if __name__ == "__main__":
    # task1()
    # task2()
    # task3()
    task4()
