from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from contrast.models.model import ModelOutput

if TYPE_CHECKING:
    from contrast.runtime.precision import PrecisionManager


@dataclass(frozen=True)
class ObjectiveMetadata:
    labels: torch.Tensor
    source_ids: torch.Tensor
    view_ids: torch.Tensor


@dataclass
class ObjectiveResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


class Objective(nn.Module):
    def initialize(self, model: nn.Module, *, total_steps: int) -> None:
        """Bind optional stateful components after the online model is on-device."""
        del model, total_steps

    def prepare_context(
        self,
        images: torch.Tensor,
        metadata: ObjectiveMetadata,
        precision: PrecisionManager,
        *,
        chunk_size: int | None,
    ) -> Any:
        """Prepare detached targets or other full-batch state before online forward."""
        del images, metadata, precision, chunk_size
        return None

    def compute(
        self,
        output: ModelOutput,
        metadata: ObjectiveMetadata,
        context: Any,
    ) -> ObjectiveResult:
        if context is not None:
            raise ValueError(f"{type(self).__name__} does not accept objective context")
        return self(output, metadata)

    def after_optimizer_step(
        self,
        model: nn.Module,
        optimizer_step: int,
    ) -> dict[str, torch.Tensor]:
        """Update non-gradient state such as target encoders and queues."""
        del model, optimizer_step
        return {}

    def forward(
        self,
        output: ModelOutput,
        metadata: ObjectiveMetadata,
    ) -> ObjectiveResult:
        raise NotImplementedError
