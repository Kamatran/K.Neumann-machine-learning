"""W4: checks of the trainer (Z1) and of TabularDataset (Z4).

Run from the project root (no running stack, no dataset needed):
    uv run pytest exercises/w04
"""

import pytest
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from mlproject.training.trainer import evaluate, train_one_epoch
from tabular import Standardize, TabularDataset


@pytest.fixture
def linear_problem():
    torch.manual_seed(0)
    X = torch.randn(100, 3)
    y = X @ torch.tensor([[1.0], [-2.0], [0.5]]) + 0.1 * torch.randn(100, 1)
    return TensorDataset(X, y)


def test_evaluate_is_the_mean_over_examples_and_changes_nothing(linear_problem):
    torch.manual_seed(1)
    model = torch.nn.Linear(3, 1)
    before = [p.clone() for p in model.parameters()]
    X, y = linear_problem.tensors
    expected = torch.nn.functional.mse_loss(model(X), y).item()
    # batch size 30: the last batch has 10 examples, so a plain mean of batch losses would be wrong
    assert evaluate(model, DataLoader(linear_problem, batch_size=30), torch.nn.MSELoss()) == pytest.approx(expected, rel=1e-5)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all(p.grad is None for p in model.parameters())


def test_train_one_epoch_learns(linear_problem):
    torch.manual_seed(1)
    model = torch.nn.Linear(3, 1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader(linear_problem, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(0))
    full = DataLoader(linear_problem, batch_size=100)
    start = evaluate(model, full, loss_fn)
    losses = [train_one_epoch(model, loader, loss_fn, optimizer) for _ in range(20)]
    assert evaluate(model, full, loss_fn) < 0.05 < start
    assert losses[-1] < losses[0]


def test_tabular_dataset():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 10.0, 30.0, 30.0], "t": [0.5, 1.5, 2.5, 3.5]})
    ds = TabularDataset(frame, ["a", "b"], "t")
    assert len(ds) == 4
    x, y = ds[2]
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert x.tolist() == [3.0, 30.0] and y.shape == (1,) and y.item() == 2.5


def test_standardize_uses_the_given_part_and_the_loader_matches_tensor_dataset():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 10.0, 30.0, 30.0], "t": [0.5, 1.5, 2.5, 3.5]})
    standardize = Standardize(frame, ["a", "b"])
    x, _ = TabularDataset(frame, ["a", "b"], "t", standardize)[0]
    assert x.tolist() == pytest.approx([-1.3416, -1.0], abs=1e-4)   # population std, as StandardScaler
    own = TabularDataset(frame, ["a", "b"], "t", standardize)
    ref = TensorDataset(standardize(torch.tensor(frame[["a", "b"]].to_numpy(), dtype=torch.float32)),
                        torch.tensor(frame["t"].to_numpy(), dtype=torch.float32).unsqueeze(1))
    batches = lambda ds: list(DataLoader(ds, batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(0)))
    for (x1, y1), (x2, y2) in zip(batches(own), batches(ref)):
        assert torch.equal(x1, x2) and torch.equal(y1, y2)
