import pytest
import torch
from torch import nn

from contrast.training.trainer import _gradient_norm, _is_better


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
