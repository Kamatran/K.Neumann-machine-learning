"""Tests of SimpleCNN (W6, task Z3). Do not change. Run: uv run pytest exercises/w06"""

import torch

from mlproject.models.cnn import SimpleCNN


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def test_output_shape():
    model = SimpleCNN(n_classes=10)
    assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 10)


def test_parameter_count():
    # conv: 448 + 4 640 + 18 496; head: 4 096 * 64 + 64 = 262 208 and 64 * 10 + 10 = 650
    assert n_params(SimpleCNN(n_classes=10)) == 286_442


def test_feature_maps():
    model = SimpleCNN(n_classes=10)
    assert model.features(torch.zeros(1, 3, 64, 64)).shape == (1, 64, 8, 8)


def test_other_image_size():
    model = SimpleCNN(n_classes=5, size=128)
    assert model(torch.zeros(1, 3, 128, 128)).shape == (1, 5)
