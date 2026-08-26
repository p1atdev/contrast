import pytest
import torch
from pydantic import ValidationError
from torch import nn

from contrast.config.schema import (
    ConstantEMADecayConfig,
    CosineEMADecayConfig,
    EMAConfig,
    InversePowerEMADecayConfig,
    LinearEMADecayConfig,
)
from contrast.training.ema import ExponentialMovingAverage, decay_at_step


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        (ConstantEMADecayConfig(decay=0.95), (0.95, 0.95, 0.95)),
        (
            LinearEMADecayConfig(
                start_decay=0.9,
                end_decay=0.99,
                schedule_steps=10,
            ),
            (0.9, 0.945, 0.99),
        ),
        (
            CosineEMADecayConfig(
                start_decay=0.9,
                end_decay=0.99,
                schedule_steps=10,
            ),
            (0.9, 0.945, 0.99),
        ),
    ],
)
def test_bounded_decay_schedules(
    schedule: ConstantEMADecayConfig | LinearEMADecayConfig | CosineEMADecayConfig,
    expected: tuple[float, float, float],
) -> None:
    values = (
        decay_at_step(schedule, 0),
        decay_at_step(schedule, 5),
        decay_at_step(schedule, 20),
    )
    assert values == pytest.approx(expected)


def test_inverse_power_schedule_is_monotonic_and_capped() -> None:
    schedule = InversePowerEMADecayConfig(
        min_decay=0.1,
        max_decay=0.99,
        inv_gamma=1.0,
        power=1.0,
    )
    values = [decay_at_step(schedule, step) for step in (0, 1, 9, 999)]

    assert values == sorted(values)
    assert values[0] == pytest.approx(0.1)
    assert values[-1] == pytest.approx(0.99)


@pytest.mark.parametrize(
    "schedule",
    [
        LinearEMADecayConfig,
        CosineEMADecayConfig,
    ],
)
def test_interpolated_schedule_rejects_reversed_range(schedule: type) -> None:
    with pytest.raises(ValidationError):
        schedule(start_decay=0.99, end_decay=0.9)


class BufferedScalar(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value]))
        self.register_buffer("running", torch.tensor([value]))
        self.register_buffer("batches", torch.tensor(0, dtype=torch.long))


def test_ema_first_update_copies_then_averages_parameters() -> None:
    source = BufferedScalar(0.0)
    ema = ExponentialMovingAverage(
        source,
        EMAConfig(
            enabled=True,
            decay=ConstantEMADecayConfig(decay=0.5),
        ),
    )

    source.weight.data.fill_(2.0)
    assert ema.update(source, optimizer_step=1)
    torch.testing.assert_close(ema.model.weight, torch.tensor([2.0]))
    assert ema.decay == 0.0

    source.weight.data.fill_(4.0)
    assert ema.update(source, optimizer_step=2)
    torch.testing.assert_close(ema.model.weight, torch.tensor([3.0]))
    assert ema.decay == 0.5
    assert ema.updates == 2
    assert not ema.model.weight.requires_grad
    assert not ema.model.training


def test_ema_start_step_and_update_interval_use_optimizer_steps() -> None:
    source = BufferedScalar(1.0)
    ema = ExponentialMovingAverage(
        source,
        EMAConfig(
            enabled=True,
            start_step=3,
            update_every_steps=2,
            decay=ConstantEMADecayConfig(decay=0.5),
        ),
    )

    assert not ema.update(source, optimizer_step=1)
    assert not ema.update(source, optimizer_step=2)
    assert ema.update(source, optimizer_step=3)
    assert not ema.update(source, optimizer_step=4)
    assert ema.update(source, optimizer_step=5)
    assert ema.updates == 2


@pytest.mark.parametrize(
    ("buffer_mode", "expected"),
    [("copy", 4.0), ("ema", 3.0)],
)
def test_ema_buffer_modes(buffer_mode: str, expected: float) -> None:
    source = BufferedScalar(0.0)
    ema = ExponentialMovingAverage(
        source,
        EMAConfig(
            enabled=True,
            buffer_mode=buffer_mode,
            decay=ConstantEMADecayConfig(decay=0.5),
        ),
    )
    source.running.fill_(2.0)
    source.batches.fill_(1)
    ema.update(source, optimizer_step=1)
    source.running.fill_(4.0)
    source.batches.fill_(2)
    ema.update(source, optimizer_step=2)

    torch.testing.assert_close(ema.model.running, torch.tensor([expected]))
    torch.testing.assert_close(ema.model.batches, torch.tensor(2))


def test_ema_state_dict_restores_shadow_and_schedule_state() -> None:
    source = BufferedScalar(0.0)
    config = EMAConfig(
        enabled=True,
        decay=ConstantEMADecayConfig(decay=0.75),
    )
    original = ExponentialMovingAverage(source, config)
    source.weight.data.fill_(2.0)
    original.update(source, optimizer_step=1)
    source.weight.data.fill_(6.0)
    original.update(source, optimizer_step=2)

    restored = ExponentialMovingAverage(BufferedScalar(-1.0), config)
    restored.load_state_dict(original.state_dict())

    torch.testing.assert_close(restored.model.weight, original.model.weight)
    assert restored.updates == original.updates
    assert restored.decay == original.decay
