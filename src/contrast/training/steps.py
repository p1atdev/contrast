from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

from contrast.models.model import ModelOutput
from contrast.objectives.base import Objective, ObjectiveMetadata, ObjectiveResult
from contrast.runtime.precision import PrecisionManager


@dataclass(frozen=True)
class PreparedBatch:
    images: torch.Tensor
    metadata: ObjectiveMetadata


def prepare_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> PreparedBatch:
    views, labels, source_ids = batch
    source_count, view_count = views.shape[:2]
    images = views.flatten(0, 1).to(device, non_blocking=True)
    metadata = ObjectiveMetadata(
        labels=labels.repeat_interleave(view_count).to(device, non_blocking=True),
        source_ids=source_ids.repeat_interleave(view_count).to(device, non_blocking=True),
        view_ids=torch.arange(view_count, device=device).repeat(source_count),
    )
    return PreparedBatch(images, metadata)


@dataclass
class RNGSnapshot:
    cpu: torch.Tensor
    cuda: torch.Tensor | None

    @classmethod
    def capture(cls, device: torch.device) -> RNGSnapshot:
        cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        return cls(torch.random.get_rng_state(), cuda)

    def restore(self, device: torch.device) -> None:
        torch.random.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state(self.cuda, device)


class DirectStep:
    def __init__(self, precision: PrecisionManager) -> None:
        self.precision = precision

    def backward(
        self,
        model: nn.Module,
        objective: Objective,
        batch: PreparedBatch,
    ) -> ObjectiveResult:
        with self.precision.autocast():
            output = model(batch.images)
        result = objective(output, batch.metadata)
        if not bool(torch.isfinite(result.loss.detach())):
            raise FloatingPointError("objective returned a non-finite loss")
        if self.precision.uses_scaler:
            self.precision.scaler.scale(result.loss).backward()
        else:
            result.loss.backward()
        return result


class GradCacheStep:
    def __init__(self, precision: PrecisionManager, chunk_size: int) -> None:
        if precision.uses_scaler:
            raise ValueError(
                "GradCache currently supports FP32 and BF16 autocast, not FP16 scaling"
            )
        self.precision = precision
        self.chunk_size = chunk_size

    def _chunks(self, images: torch.Tensor) -> Iterator[torch.Tensor]:
        yield from images.split(self.chunk_size)

    def backward(
        self,
        model: nn.Module,
        objective: Objective,
        batch: PreparedBatch,
    ) -> ObjectiveResult:
        snapshots: list[RNGSnapshot] = []
        cached: list[ModelOutput] = []
        device = batch.images.device
        with torch.no_grad():
            for images in self._chunks(batch.images):
                snapshots.append(RNGSnapshot.capture(device))
                with self.precision.autocast():
                    cached.append(model(images))

        leaves = ModelOutput(
            features=torch.cat([item.features for item in cached]).detach().requires_grad_(),
            embeddings=torch.cat([item.embeddings for item in cached]).detach().requires_grad_(),
            logits=torch.cat([item.logits for item in cached]).detach().requires_grad_(),
        )
        result = objective(leaves, batch.metadata)
        if not bool(torch.isfinite(result.loss.detach())):
            raise FloatingPointError("objective returned a non-finite loss")
        result.loss.backward()

        gradients = {
            name: value.grad.split(self.chunk_size)
            for name, value in leaves.as_dict().items()
            if value.grad is not None
        }
        cuda_devices = [device.index or 0] if device.type == "cuda" else []
        for index, images in enumerate(self._chunks(batch.images)):
            with torch.random.fork_rng(devices=cuda_devices):
                snapshots[index].restore(device)
                with self.precision.autocast():
                    output = model(images)
                surrogate = sum(
                    (value * gradients[name][index]).sum()
                    for name, value in output.as_dict().items()
                    if name in gradients
                )
                surrogate.backward()
        return result
