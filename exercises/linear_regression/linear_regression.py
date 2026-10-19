"""W1 homework, part 1: two ways to the same weights (California Housing; the lecture demo).

Fill in the three functions marked TODO and the loop at the end of main().
Check your functions first:
    uv run pytest exercises/linear_regression
Then run the experiment from the project root:
    uv run python exercises/linear_regression/linear_regression.py
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from mlproject.tracking import setup

SEED = 42
LEARNING_RATES = (0.01, 0.1, 0.5)
MAX_ITER = 20_000
TOL = 1e-6  # stop once the training MSE is less than TOL above the normal-equation MSE
DIVERGED_ABOVE = 1e6  # a training MSE above this value means the run diverged


def load_split():
    """California Housing, 80/20 split with the course seed. Downloads the data on first use."""
    X, y = fetch_california_housing(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


def add_intercept(X):
    """Prepends a column of ones, so that w[0] is the intercept."""
    return np.c_[np.ones(len(X)), X]


def mse(y, pred):
    return float(np.mean((y - pred) ** 2))


def mae(y, pred):
    return float(np.mean(np.abs(y - pred)))


def standardize(X_train, X_test):
    """Scales both parts with the mean and standard deviation of the training part only.

    Returns (Z_train, Z_test). Use the population standard deviation (NumPy's default),
    as StandardScaler does.
    """
    # TODO (homework): compute the statistics on X_train only and apply them to both parts.
    raise NotImplementedError("homework: standardize")


def fit_normal_equation(A, y):
    """Weights minimising the MSE: solve A^T A w = A^T y.

    A already contains the column of ones. Solve the linear system (np.linalg.solve),
    do not invert the matrix.
    """
    # TODO (homework)
    raise NotImplementedError("homework: fit_normal_equation")


def fit_gradient_descent(A, y, lr, max_iter, tol, target_mse):
    """Gradient descent on the MSE, starting from zero weights.

    One step: gradient (2/n) A^T (A w - y), then w <- w - lr * gradient.
    After every step compute the training MSE and append it to losses.

    Returns (w, losses, status). losses[k] is the training MSE after step k + 1.
    status is:
      "diverged"  - the MSE is not finite or above DIVERGED_ABOVE (check this first),
      "converged" - the MSE is less than tol above target_mse,
      "max_iter"  - neither happened within max_iter steps.
    """
    # TODO (homework)
    raise NotImplementedError("homework: fit_gradient_descent")


def log_curve(key, values):
    """Logs the value of every step. The values travel in batches of 1000 steps, one request per batch."""
    run_id = mlflow.active_run().info.run_id
    now = int(time.time() * 1000)
    metrics = [Metric(key, float(v), now, step) for step, v in enumerate(values, start=1) if np.isfinite(v)]
    client = MlflowClient()
    for start in range(0, len(metrics), 1000):
        client.log_batch(run_id, metrics=metrics[start:start + 1000])


def excess_log10(losses, target_mse):
    """log10 of how far the training MSE is above the normal-equation MSE, for every step.

    The training MSE drops in the first few steps and then looks flat, so its chart hides convergence.
    On this scale a converging run is a falling straight line (its slope is the speed of convergence)
    and a diverging run turns upwards.
    """
    return [float(np.log10(max(loss - target_mse, 1e-12))) for loss in losses]


def plot_loss_curve(losses, target_mse, lr):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, label=f"lr = {lr}")
    ax.axhline(target_mse, color="black", ls="--", label=f"normal equation ({target_mse:.4f})")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("training MSE (log scale)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(y, pred):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y, pred, s=3, alpha=0.3)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color="black", ls="--")
    ax.set_xlabel("true value (100k USD)")
    ax.set_ylabel("prediction (100k USD)")
    fig.tight_layout()
    return fig


def main():
    setup("w01-california")
    X_train, X_test, y_train, y_test = load_split()
    Z_train, Z_test = standardize(X_train, X_test)
    A_train, A_test = add_intercept(Z_train), add_intercept(Z_test)

    with mlflow.start_run(run_name="baseline-mean"):
        pred = np.full_like(y_test, y_train.mean())
        mlflow.log_param("method", "mean of the training target")
        mlflow.log_metrics({"test_mse": mse(y_test, pred), "test_mae": mae(y_test, pred)})
    print(f"baseline-mean     test MSE {mse(y_test, pred):.4f}")

    with mlflow.start_run(run_name="normal-equation"):
        start = time.perf_counter()
        w_ne = fit_normal_equation(A_train, y_train)
        solve_ms = (time.perf_counter() - start) * 1000
        train_mse_ne = mse(y_train, A_train @ w_ne)
        pred = A_test @ w_ne
        mlflow.log_param("method", "normal equation")
        mlflow.log_metrics({"train_mse": train_mse_ne, "test_mse": mse(y_test, pred),
                            "test_mae": mae(y_test, pred), "solve_ms": solve_ms})
        fig = plot_residuals(y_test, pred)
        mlflow.log_figure(fig, "residuals.png")
        plt.close(fig)
    print(f"normal-equation   test MSE {mse(y_test, pred):.4f}, train MSE {train_mse_ne:.4f}")

    # TODO (homework): for every lr in LEARNING_RATES start a run named f"gd-lr-{lr}" and, following
    # the two runs above:
    #   1. log the parameters method, lr, max_iter and tol;
    #   2. call fit_gradient_descent with train_mse_ne as target_mse;
    #   3. log status as a tag (mlflow.set_tag) and the number of iterations as a metric;
    #   4. log the whole loss curve with log_curve("train_mse", losses), and for a readable chart in the
    #      MLflow UI also log_curve("train_excess_log10", excess_log10(losses, train_mse_ne));
    #   5. unless the run diverged, log test_mse and test_mae;
    #   6. log plot_loss_curve(...) as "loss_curve.png" and close the figure;
    #   7. print one line: run name, status, number of iterations.


if __name__ == "__main__":
    main()
