"""W1 homework: checks of your functions against scikit-learn.

Run from the project root (no running stack needed):
    uv run pytest exercises/linear_regression
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from linear_regression import add_intercept, fit_gradient_descent, fit_normal_equation, load_split, mse, standardize


@pytest.fixture(scope="module")
def data():
    # Scaled with scikit-learn here, so that each test checks exactly one of your functions.
    X_train, X_test, y_train, _ = load_split()
    scaler = StandardScaler().fit(X_train)
    Z_train, Z_test = scaler.transform(X_train), scaler.transform(X_test)
    return X_train, X_test, y_train, add_intercept(Z_train), add_intercept(Z_test), Z_train, Z_test


def test_standardize_uses_training_statistics_only(data):
    X_train, X_test, _, _, _, _, _ = data
    Z_train, Z_test = standardize(X_train, X_test)
    scaler = StandardScaler().fit(X_train)
    np.testing.assert_allclose(Z_train, scaler.transform(X_train), atol=1e-10)
    np.testing.assert_allclose(Z_test, scaler.transform(X_test), atol=1e-10)


def test_normal_equation_matches_linear_regression(data):
    _, _, y_train, A_train, A_test, Z_train, Z_test = data
    w = fit_normal_equation(A_train, y_train)
    reference = LinearRegression().fit(Z_train, y_train).predict(Z_test)
    np.testing.assert_allclose(A_test @ w, reference, atol=1e-8)


def test_gradient_descent_reaches_normal_equation(data):
    _, _, y_train, A_train, _, _, _ = data
    w_ne = fit_normal_equation(A_train, y_train)
    target = mse(y_train, A_train @ w_ne)
    w, losses, status = fit_gradient_descent(A_train, y_train, lr=0.1, max_iter=20_000, tol=1e-6, target_mse=target)
    assert status == "converged"
    assert losses[-1] - target < 1e-6
    assert len(losses) < 20_000
    np.testing.assert_allclose(w, w_ne, atol=0.05)


def test_gradient_descent_reports_divergence(data):
    _, _, y_train, A_train, _, _, _ = data
    w, losses, status = fit_gradient_descent(A_train, y_train, lr=1.1, max_iter=20_000, tol=1e-6, target_mse=0.0)
    assert status == "diverged"
    assert len(losses) < 100
