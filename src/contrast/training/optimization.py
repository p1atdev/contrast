from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

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
        self,
        config: OptimizerConfig,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        steps: int,
    ) -> None:
        self.config = config
        self.scheduler: LambdaLR | None = None
        parameters = [
            (name, parameter) for name, parameter in named_parameters if parameter.requires_grad
        ]
        self._source_indices: list[list[int]] = []
        optimizer_parameters: list[dict[str, Any]]
        if config.weight_decay_policy == "all":
            self._source_indices = [list(range(len(parameters)))]
            optimizer_parameters = [
                {
                    "params": [parameter for _, parameter in parameters],
                    "weight_decay": config.weight_decay,
                }
            ]
        else:
            decay_indices = [
                index
                for index, (_, parameter) in enumerate(parameters)
                if parameter.ndim > 1 and not bool(getattr(parameter, "_no_weight_decay", False))
            ]
            decay_index_set = set(decay_indices)
            no_decay_indices = [
                index for index in range(len(parameters)) if index not in decay_index_set
            ]
            self._source_indices = [decay_indices, no_decay_indices]
            optimizer_parameters = [
                {
                    "params": [parameters[index][1] for index in decay_indices],
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": [parameters[index][1] for index in no_decay_indices],
                    "weight_decay": 0.0,
                },
            ]
        if isinstance(config, AdamWScheduleFreeOptimizerConfig):
            self.optimizer: Optimizer = AdamWScheduleFree(
                optimizer_parameters,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                weight_decay=0.0,
                warmup_steps=config.warmup_steps,
            )
            self.schedule_free = True
        elif isinstance(config, AdamWOptimizerConfig):
            self.optimizer = torch.optim.AdamW(
                optimizer_parameters,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                weight_decay=0.0,
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
        optimizer_state = state["optimizer"]
        saved_groups = optimizer_state["param_groups"]
        if len(saved_groups) == 1 and len(self.optimizer.param_groups) > 1:
            saved_group = saved_groups[0]
            saved_parameters = saved_group["params"]
            if len(saved_parameters) != sum(map(len, self._source_indices)):
                raise ValueError("legacy optimizer state has an unexpected parameter count")
            migrated_groups = []
            for current_group, source_indices in zip(
                self.optimizer.param_groups,
                self._source_indices,
                strict=True,
            ):
                migrated_group = {
                    key: value for key, value in saved_group.items() if key != "params"
                }
                migrated_group["params"] = [saved_parameters[index] for index in source_indices]
                migrated_group["weight_decay"] = current_group["weight_decay"]
                migrated_groups.append(migrated_group)
            optimizer_state = {
                **optimizer_state,
                "param_groups": migrated_groups,
            }
        self.optimizer.load_state_dict(optimizer_state)
        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])

    @property
    def lr(self) -> float:
        group = self.optimizer.param_groups[0]
        if self.schedule_free:
            return float(group["scheduled_lr"])
        return float(group["lr"])
