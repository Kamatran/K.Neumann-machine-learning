"""W5: checks of the MLP class (Z1).

Run from the project root (no stack, no dataset needed):
    uv run pytest exercises/w05
"""

import pytest
import torch
from torch import nn

from mlproject.evaluation.classification import confusion
from mlproject.models.mlp import MLP, count_parameters


def test_output_shape_for_images():
    model = MLP(3 * 64 * 64, 10, hidden=(256,))
    assert model(torch.zeros(4, 3, 64, 64)).shape == (4, 10)


def test_parameter_count():
    # 12288 * 256 + 256 weights and biases of the hidden layer, 256 * 10 + 10 of the output layer
    assert count_parameters(MLP(3 * 64 * 64, 10, hidden=(256,))) == 3_148_554
    assert count_parameters(MLP(100, 5, hidden=(64, 32))) == 100 * 64 + 64 + 64 * 32 + 32 + 32 * 5 + 5


def test_layers_and_activation():
    layers = list(MLP(10, 3, hidden=(8, 8), activation=nn.Tanh).net)
    assert [type(l) for l in layers] == [nn.Flatten, nn.Linear, nn.Tanh, nn.Linear, nn.Tanh, nn.Linear]


@pytest.mark.parametrize("activation, collapses", [(None, True), (nn.ReLU, False)])
def test_without_activation_depth_collapses_to_an_affine_map(activation, collapses):
    # An affine map f satisfies f(a*x1 + b*x2) = a*f(x1) + b*f(x2) when a + b = 1.
    torch.manual_seed(0)
    f = MLP(20, 4, hidden=(32, 32, 32), activation=activation).double()
    x1, x2 = torch.randn(1, 20, dtype=torch.float64), torch.randn(1, 20, dtype=torch.float64)
    gap = (f(0.3 * x1 + 0.7 * x2) - (0.3 * f(x1) + 0.7 * f(x2))).abs().max().item()
    assert (gap < 1e-12) == collapses


def test_confusion():
    cm = confusion(torch.tensor([0, 1, 1, 2]).numpy(), torch.tensor([0, 2, 1, 2]).numpy(), 3)
    assert cm.tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
