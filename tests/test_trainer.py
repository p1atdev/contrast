from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from contrast.training.trainer import Trainer, _gradient_norm, _is_better


def test_gradient_norm_combines_parameter_gradients() -> None:
    first = nn.Parameter(torch.zeros(2))
    second = nn.Parameter(torch.zeros(1))
    unused = nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([12.0])

    norm = _gradient_norm((first, second, unused))

    assert float(norm) == pytest.approx(13.0)


@pytest.mark.parametrize(
    ("value", "best", "mode", "expected"),
    [
        (0.5, None, "max", True),
        (0.6, 0.5, "max", True),
        (0.4, 0.5, "max", False),
        (0.4, 0.5, "min", True),
        (0.6, 0.5, "min", False),
    ],
)
def test_is_better(value: float, best: float | None, mode: str, expected: bool) -> None:
    assert _is_better(value, best, mode) is expected


def test_fit_treats_configured_max_steps_as_success(monkeypatch, tmp_path: Path) -> None:
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(
            epochs=3,
            max_steps=2,
            evaluate_every_epochs=1,
            checkpoint_every_epochs=1,
        ),
        evaluation=SimpleNamespace(enabled=False),
    )
    trainer.epoch = 0
    trainer.global_step = 2
    logged = []
    trainer.store = SimpleNamespace(log=logged.append)
    monkeypatch.setattr(trainer, "_train_epoch", lambda: False)
    monkeypatch.setattr(trainer, "_checkpoint", lambda _name: tmp_path / "final.pt")

    result = trainer.fit()

    assert result == tmp_path / "final.pt"
    assert logged[-1]["completed"] is True
    assert logged[-1]["stopped_at_max_steps"] is True
