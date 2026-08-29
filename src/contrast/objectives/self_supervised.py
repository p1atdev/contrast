from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from contrast.models.model import ModelOutput
from contrast.objectives.base import Objective, ObjectiveMetadata, ObjectiveResult
from contrast.runtime.precision import PrecisionManager


def _raw_embeddings(output: ModelOutput) -> torch.Tensor:
    return output.raw_embeddings if output.raw_embeddings is not None else output.embeddings


def _two_views(
    values: torch.Tensor,
    metadata: ObjectiveMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_mask = metadata.view_ids == 0
    second_mask = metadata.view_ids == 1
    if int(first_mask.sum()) != int(second_mask.sum()) or not bool(first_mask.any()):
        raise ValueError("objective requires exactly two equally sized views")
    if bool(((metadata.view_ids != 0) & (metadata.view_ids != 1)).any()):
        raise ValueError("objective requires exactly two views")
    first_sources = metadata.source_ids[first_mask]
    second_sources = metadata.source_ids[second_mask]
    if not torch.equal(first_sources, second_sources):
        raise ValueError("objective views are not aligned by source")
    return values[first_mask], values[second_mask]


class BarlowTwinsObjective(Objective):
    def __init__(self, redundancy_weight: float, eps: float) -> None:
        super().__init__()
        self.redundancy_weight = redundancy_weight
        self.eps = eps

    def forward(self, output: ModelOutput, metadata: ObjectiveMetadata) -> ObjectiveResult:
        first, second = _two_views(_raw_embeddings(output).float(), metadata)
        first = (first - first.mean(dim=0)) * torch.rsqrt(
            first.var(dim=0, unbiased=False) + self.eps
        )
        second = (second - second.mean(dim=0)) * torch.rsqrt(
            second.var(dim=0, unbiased=False) + self.eps
        )
        correlation = first.T @ second / first.shape[0]
        diagonal = torch.diagonal(correlation)
        on_diagonal = (diagonal - 1.0).square().sum()
        off_diagonal_mask = ~torch.eye(
            correlation.shape[0],
            dtype=torch.bool,
            device=correlation.device,
        )
        off_diagonal = correlation.masked_select(off_diagonal_mask).square().sum()
        loss = on_diagonal + self.redundancy_weight * off_diagonal
        return ObjectiveResult(
            loss,
            {
                "loss/barlow_twins": loss.detach(),
                "barlow_twins/on_diagonal": on_diagonal.detach(),
                "barlow_twins/off_diagonal": off_diagonal.detach(),
                "barlow_twins/diagonal_mean": diagonal.mean().detach(),
            },
        )


@dataclass(frozen=True)
class _TargetContext:
    embeddings: torch.Tensor
    raw_embeddings: torch.Tensor


class _MomentumObjective(Objective):
    def __init__(self) -> None:
        super().__init__()
        self.target_model: nn.Module | None = None
        self.total_steps = 1
        self.register_buffer("target_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_target_decay", torch.zeros((), dtype=torch.float32))

    def initialize(self, model: nn.Module, *, total_steps: int) -> None:
        if self.target_model is not None:
            raise RuntimeError("momentum objective was already initialized")
        self.target_model = copy.deepcopy(model)
        self.target_model.requires_grad_(False)
        self.target_model.eval()
        self.total_steps = max(1, total_steps)

    def train(self, mode: bool = True) -> _MomentumObjective:
        super().train(mode)
        if self.target_model is not None:
            self.target_model.eval()
        return self

    def prepare_context(
        self,
        images: torch.Tensor,
        metadata: ObjectiveMetadata,
        precision: PrecisionManager,
        *,
        chunk_size: int | None,
    ) -> _TargetContext:
        del metadata
        if self.target_model is None:
            raise RuntimeError("momentum objective is not initialized")
        size = chunk_size or images.shape[0]
        embeddings: list[torch.Tensor] = []
        raw_embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for chunk in images.split(size):
                with precision.autocast():
                    output = self.target_model(chunk)
                if not isinstance(output, ModelOutput):
                    raise TypeError("target model must return ModelOutput")
                embeddings.append(output.embeddings.float())
                raw_embeddings.append(_raw_embeddings(output).float())
        return _TargetContext(
            embeddings=torch.cat(embeddings),
            raw_embeddings=torch.cat(raw_embeddings),
        )

    def target_decay(self, optimizer_step: int) -> float:
        raise NotImplementedError

    @torch.no_grad()
    def _update_target(self, model: nn.Module, decay: float) -> None:
        if self.target_model is None:
            raise RuntimeError("momentum objective is not initialized")
        target_parameters = self.target_model.named_parameters()
        online_parameters = model.named_parameters()
        for (target_name, target), (online_name, online) in zip(
            target_parameters,
            online_parameters,
            strict=True,
        ):
            if target_name != online_name:
                raise ValueError(f"target parameter mismatch: {target_name} != {online_name}")
            target.lerp_(online.detach(), 1.0 - decay)
        target_buffers = self.target_model.named_buffers()
        online_buffers = model.named_buffers()
        for (target_name, target), (online_name, online) in zip(
            target_buffers,
            online_buffers,
            strict=True,
        ):
            if target_name != online_name:
                raise ValueError(f"target buffer mismatch: {target_name} != {online_name}")
            target.copy_(online.detach())
        self.target_updates.add_(1)
        self.last_target_decay.fill_(decay)
        self.target_model.eval()

    def after_optimizer_step(
        self,
        model: nn.Module,
        optimizer_step: int,
    ) -> dict[str, torch.Tensor]:
        decay = self.target_decay(optimizer_step)
        self._update_target(model, decay)
        return {
            "target/decay": self.last_target_decay.detach(),
            "target/updates": self.target_updates.detach().float(),
        }


class BYOLObjective(_MomentumObjective):
    def __init__(
        self,
        embedding_dim: int,
        predictor_hidden_dim: int,
        base_target_decay: float,
        final_target_decay: float,
    ) -> None:
        super().__init__()
        self.base_target_decay = base_target_decay
        self.final_target_decay = final_target_decay
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, embedding_dim),
        )

    def target_decay(self, optimizer_step: int) -> float:
        progress = min(max(optimizer_step, 0) / self.total_steps, 1.0)
        interpolation = 0.5 * (1.0 - math.cos(math.pi * progress))
        return (
            self.base_target_decay
            + (self.final_target_decay - self.base_target_decay) * interpolation
        )

    def compute(
        self,
        output: ModelOutput,
        metadata: ObjectiveMetadata,
        context: _TargetContext,
    ) -> ObjectiveResult:
        if not isinstance(context, _TargetContext):
            raise TypeError("BYOL requires momentum target context")
        online = F.normalize(self.predictor(_raw_embeddings(output).float()), dim=-1)
        target = F.normalize(context.raw_embeddings.detach(), dim=-1)
        online_first, online_second = _two_views(online, metadata)
        target_first, target_second = _two_views(target, metadata)
        cosine_first = (online_first * target_second).sum(dim=1)
        cosine_second = (online_second * target_first).sum(dim=1)
        cosine = torch.cat((cosine_first, cosine_second))
        loss = (2.0 - 2.0 * cosine).mean()
        return ObjectiveResult(
            loss,
            {
                "loss/byol": loss.detach(),
                "byol/cosine_similarity": cosine.mean().detach(),
                "byol/target_decay": self.last_target_decay.detach(),
            },
        )


class MoCoObjective(_MomentumObjective):
    def __init__(
        self,
        embedding_dim: int,
        queue_size: int,
        temperature: float,
        target_decay: float,
        symmetric: bool,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.fixed_target_decay = target_decay
        self.symmetric = symmetric
        queue = F.normalize(torch.randn(queue_size, embedding_dim), dim=1)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_pointer", torch.zeros((), dtype=torch.long))
        self._pending_keys: torch.Tensor | None = None

    def target_decay(self, optimizer_step: int) -> float:
        del optimizer_step
        return self.fixed_target_decay

    def compute(
        self,
        output: ModelOutput,
        metadata: ObjectiveMetadata,
        context: _TargetContext,
    ) -> ObjectiveResult:
        if not isinstance(context, _TargetContext):
            raise TypeError("MoCo requires momentum target context")
        if self._pending_keys is not None:
            raise RuntimeError("MoCo queue update from the previous step is still pending")
        online_first, online_second = _two_views(output.embeddings.float(), metadata)
        target_first, target_second = _two_views(context.embeddings.detach(), metadata)
        if self.symmetric:
            queries = torch.cat((online_first, online_second))
            keys = torch.cat((target_second, target_first))
            enqueue_keys = torch.cat((target_first, target_second))
        else:
            queries = online_first
            keys = target_second
            enqueue_keys = target_second
        queries = F.normalize(queries, dim=-1)
        keys = F.normalize(keys, dim=-1)
        positive_logits = (queries * keys).sum(dim=1, keepdim=True)
        negative_logits = queries @ self.queue.detach().T
        logits = torch.cat((positive_logits, negative_logits), dim=1) / self.temperature
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, targets)
        self._pending_keys = F.normalize(enqueue_keys.detach(), dim=-1)
        return ObjectiveResult(
            loss,
            {
                "loss/moco": loss.detach(),
                "moco/positive_similarity": positive_logits.mean().detach(),
                "moco/negative_similarity": negative_logits.mean().detach(),
                "moco/queue_pointer": self.queue_pointer.detach().float(),
                "moco/target_decay": self.last_target_decay.detach(),
            },
        )

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor) -> None:
        queue_size = self.queue.shape[0]
        if keys.shape[0] >= queue_size:
            self.queue.copy_(keys[-queue_size:])
            self.queue_pointer.zero_()
            return
        pointer = int(self.queue_pointer)
        first_count = min(keys.shape[0], queue_size - pointer)
        self.queue[pointer : pointer + first_count].copy_(keys[:first_count])
        remaining = keys.shape[0] - first_count
        if remaining:
            self.queue[:remaining].copy_(keys[first_count:])
        self.queue_pointer.fill_((pointer + keys.shape[0]) % queue_size)

    def after_optimizer_step(
        self,
        model: nn.Module,
        optimizer_step: int,
    ) -> dict[str, torch.Tensor]:
        if self._pending_keys is None:
            raise RuntimeError("MoCo optimizer step has no pending queue keys")
        metrics = super().after_optimizer_step(model, optimizer_step)
        self._enqueue(self._pending_keys)
        self._pending_keys = None
        metrics.update(
            {
                "moco/queue_pointer": self.queue_pointer.detach().float(),
                "moco/target_decay": self.last_target_decay.detach(),
            }
        )
        return metrics
