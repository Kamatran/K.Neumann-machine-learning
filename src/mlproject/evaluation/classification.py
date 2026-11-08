"""Multi-class evaluation helpers (W5). Complete, do not change."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def predict(model, loader):
    """True and predicted labels for all examples of loader, as NumPy arrays."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            y_true.append(np.asarray(y))
            y_pred.append(model(x).argmax(dim=1).numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def confusion(y_true, y_pred, n_classes):
    """Confusion matrix: rows are true classes, columns predicted classes."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def confusion_figure(cm, class_names):
    fig, ax = plt.subplots(figsize=(0.6 * len(class_names) + 3, 0.6 * len(class_names) + 2.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(class_names)), class_names, rotation=90)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    return fig
