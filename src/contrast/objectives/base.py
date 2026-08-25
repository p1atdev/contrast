from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from contrast.models.model import ModelOutput


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
    def forward(
        self,
        output: ModelOutput,
        metadata: ObjectiveMetadata,
    ) -> ObjectiveResult:
        raise NotImplementedError
