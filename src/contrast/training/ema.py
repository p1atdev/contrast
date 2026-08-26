from __future__ import annotations

import copy
import math

import torch
from torch import nn

from contrast.config.schema import (
    ConstantEMADecayConfig,
    CosineEMADecayConfig,
    EMAConfig,
    EMADecayConfig,
    InversePowerEMADecayConfig,
    LinearEMADecayConfig,
)


def decay_at_step(config: EMADecayConfig, step: int) -> float:
    schedule_step = max(0, step)
    if isinstance(config, ConstantEMADecayConfig):
        return config.decay
    if isinstance(config, (LinearEMADecayConfig, CosineEMADecayConfig)):
        progress = min(schedule_step / config.schedule_steps, 1.0)
        if isinstance(config, CosineEMADecayConfig):
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        return config.start_decay + (config.end_decay - config.start_decay) * progress
    if isinstance(config, InversePowerEMADecayConfig):
        decay = 1.0 - (1.0 + schedule_step / config.inv_gamma) ** (-config.power)
        return min(config.max_decay, max(config.min_decay, decay))
    raise TypeError(f"unsupported EMA decay config: {type(config).__name__}")


class ExponentialMovingAverage(nn.Module):
    def __init__(self, source: nn.Module, config: EMAConfig) -> None:
        super().__init__()
        if not config.enabled:
            raise ValueError("cannot construct EMA from a disabled config")
        self.config = config
        self.model = copy.deepcopy(source)
        self.model.requires_grad_(False)
        self.model.eval()
        reference = next(source.parameters(), None)
        device = reference.device if reference is not None else torch.device("cpu")
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long, device=device))
        self.register_buffer("last_decay", torch.zeros((), dtype=torch.float32, device=device))

    @staticmethod
    @torch.no_grad()
    def _copy_parameters(target: nn.Module, source: nn.Module) -> None:
        target_parameters = target.named_parameters()
        source_parameters = source.named_parameters()
        for (target_name, target_parameter), (source_name, source_parameter) in zip(
            target_parameters,
            source_parameters,
            strict=True,
        ):
            if target_name != source_name:
                raise ValueError(f"EMA parameter mismatch: {target_name} != {source_name}")
            target_parameter.copy_(source_parameter.detach())

    @torch.no_grad()
    def _update_parameters(self, source: nn.Module, decay: float) -> None:
        for (target_name, target_parameter), (source_name, source_parameter) in zip(
            self.model.named_parameters(),
            source.named_parameters(),
            strict=True,
        ):
            if target_name != source_name:
                raise ValueError(f"EMA parameter mismatch: {target_name} != {source_name}")
            target_parameter.lerp_(source_parameter.detach(), 1.0 - decay)

    @torch.no_grad()
    def _update_buffers(self, source: nn.Module, decay: float) -> None:
        for (target_name, target_buffer), (source_name, source_buffer) in zip(
            self.model.named_buffers(),
            source.named_buffers(),
            strict=True,
        ):
            if target_name != source_name:
                raise ValueError(f"EMA buffer mismatch: {target_name} != {source_name}")
            if self.config.buffer_mode == "ema" and (
                target_buffer.is_floating_point() or target_buffer.is_complex()
            ):
                target_buffer.lerp_(source_buffer.detach(), 1.0 - decay)
            else:
                target_buffer.copy_(source_buffer.detach())

    @torch.no_grad()
    def reset(self, source: nn.Module) -> None:
        self._copy_parameters(self.model, source)
        self._update_buffers(source, decay=0.0)
        self.num_updates.zero_()
        self.last_decay.zero_()
        self.model.eval()

    @torch.no_grad()
    def update(self, source: nn.Module, optimizer_step: int) -> bool:
        activation_step = max(1, self.config.start_step)
        if optimizer_step < activation_step:
            return False
        elapsed_steps = optimizer_step - activation_step
        if elapsed_steps % self.config.update_every_steps:
            return False

        if int(self.num_updates) == 0:
            decay = 0.0
            self._copy_parameters(self.model, source)
        else:
            decay = decay_at_step(self.config.decay, elapsed_steps)
            self._update_parameters(source, decay)
        self._update_buffers(source, decay)
        self.num_updates.add_(1)
        self.last_decay.fill_(decay)
        self.model.eval()
        return True

    @property
    def updates(self) -> int:
        return int(self.num_updates)

    @property
    def decay(self) -> float:
        return float(self.last_decay)
