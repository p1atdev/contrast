from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from schedulefree import AdamWScheduleFree
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from contrast.config.schema import (
    AdamWOptimizerConfig,
    AdamWScheduleFreeOptimizerConfig,
    OptimizerConfig,
)


class OptimizationController:
    def __init__(
        self, config: OptimizerConfig, parameters: Iterator[nn.Parameter], steps: int
    ) -> None:
        self.config = config
        self.scheduler: LambdaLR | None = None
        if isinstance(config, AdamWScheduleFreeOptimizerConfig):
            self.optimizer: Optimizer = AdamWScheduleFree(
                parameters,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay,
                warmup_steps=config.warmup_steps,
            )
            self.schedule_free = True
        elif isinstance(config, AdamWOptimizerConfig):
            self.optimizer = torch.optim.AdamW(
                parameters,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay,
            )
            warmup = config.scheduler.warmup_steps
            minimum = config.scheduler.minimum_lr_ratio

            def multiplier(step: int) -> float:
                if warmup and step < warmup:
                    return (step + 1) / warmup
                progress = (step - warmup) / max(1, steps - warmup)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
                return minimum + (1.0 - minimum) * cosine

            self.scheduler = LambdaLR(self.optimizer, multiplier)
            self.schedule_free = False
        else:
            raise TypeError(f"unsupported optimizer config: {type(config).__name__}")

    def train(self) -> None:
        if self.schedule_free:
            self.optimizer.train()  # type: ignore[attr-defined]

    def eval(self) -> None:
        if self.schedule_free:
            self.optimizer.eval()  # type: ignore[attr-defined]

    @contextmanager
    def evaluation_parameters(self) -> Iterator[None]:
        self.eval()
        try:
            yield
        finally:
            self.train()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, scaler: torch.amp.GradScaler | None = None) -> None:
        if scaler is not None and scaler.is_enabled():
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])
