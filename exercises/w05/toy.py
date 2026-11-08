"""W5, task Z2: two interleaved spirals and a picture of a decision boundary.
Prepared by the lecturer; do not change.

    from toy import boundary_figure, spirals
    (X_train, y_train), (X_val, y_val) = spirals()
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset


def spirals(n=600, turns=1.5, noise=0.2, seed=0):
    """Two spirals of n points in total, 60% training and 20% validation (the last 20% are never used here).

    Returns (X_train, y_train), (X_val, y_val) as NumPy arrays; X has two columns, y is 0 or 1.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    t = np.sqrt(rng.uniform(0, 1, half)) * turns * 2 * np.pi
    arm = np.c_[t * np.cos(t), t * np.sin(t)] / (turns * 2 * np.pi) * 3
    X = np.r_[arm, -arm] + rng.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(half), np.ones(half)].astype(int)
    order = rng.permutation(n)
    X, y = X[order], y[order]
    k, v = int(0.6 * n), int(0.8 * n)
    return (X[:k], y[:k]), (X[k:v], y[k:v])


def tensors(X, y):
    """float32 features and float32 labels of shape (n, 1), as BCEWithLogitsLoss expects."""
    return TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1))


def accuracy(model, X, y):
    """Share of correct answers of a model that returns one logit per example (class 1 when the logit > 0)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).ravel().numpy()
    return float(((logits > 0).astype(int) == y).mean())


def boundary_figure(model, X, y, title=""):
    """The plane coloured by the model's probability of class 1, the 0.5 boundary and the points."""
    model.eval()
    g = np.linspace(-4, 4, 300)
    gx, gy = np.meshgrid(g, g)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32))).numpy()
    p = p.reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(gx, gy, p, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
    if p.max() > 0.5 > p.min():
        ax.contour(gx, gy, p, levels=[0.5], colors="#333333", linewidths=1.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolors="#222222", s=18, vmin=0, vmax=1)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.tight_layout()
    return fig
