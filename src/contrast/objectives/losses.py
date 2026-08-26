from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from contrast.models.model import ModelOutput
from contrast.objectives.base import Objective, ObjectiveMetadata, ObjectiveResult
from contrast.objectives.masks import (
    class_positive_mask,
    instance_positive_mask,
    valid_pair_mask,
)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values.masked_select(mask)
    if not selected.numel():
        raise ValueError("objective received an empty pair set")
    return selected.mean()


def _mean_normalized_anchor_loss(
    values: torch.Tensor,
    selected: torch.Tensor,
    normalization_counts: torch.Tensor,
) -> torch.Tensor:
    active = normalization_counts > 0
    if not bool(active.any()):
        raise ValueError("objective received no active anchors")
    per_anchor = torch.where(selected, values, 0.0).sum(dim=1)
    per_anchor = per_anchor[active] / normalization_counts[active]
    return per_anchor.mean()


class CrossEntropyObjective(Objective):
    def forward(self, output: ModelOutput, metadata: ObjectiveMetadata) -> ObjectiveResult:
        loss = F.cross_entropy(output.logits.float(), metadata.labels)
        return ObjectiveResult(loss, {"loss/ce": loss.detach()})


class SoftmaxContrastiveObjective(Objective):
    def __init__(self, temperature: float, positive_kind: str) -> None:
        super().__init__()
        self.temperature = temperature
        self.positive_kind = positive_kind

    def positive_mask(self, metadata: ObjectiveMetadata) -> torch.Tensor:
        if self.positive_kind == "instance":
            return instance_positive_mask(metadata.source_ids)
        return class_positive_mask(metadata.labels)

    def forward(self, output: ModelOutput, metadata: ObjectiveMetadata) -> ObjectiveResult:
        embeddings = output.embeddings.float()
        similarities = embeddings @ embeddings.T / self.temperature
        valid = valid_pair_mask(embeddings.shape[0], embeddings.device)
        positives = self.positive_mask(metadata)
        log_denominator = torch.logsumexp(
            similarities.masked_fill(~valid, -torch.inf),
            dim=1,
        )
        log_probability = similarities - log_denominator[:, None]
        counts = positives.sum(dim=1)
        active = counts > 0
        per_anchor = -torch.where(positives, log_probability, 0.0).sum(dim=1)
        per_anchor = per_anchor[active] / counts[active]
        loss = per_anchor.mean()
        return ObjectiveResult(
            loss,
            {
                "loss/contrastive": loss.detach(),
                "pairs/positive_per_anchor": counts.float().mean().detach(),
            },
        )


class SincereObjective(Objective):
    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, output: ModelOutput, metadata: ObjectiveMetadata) -> ObjectiveResult:
        embeddings = output.embeddings.float()
        similarities = embeddings @ embeddings.T / self.temperature
        positives = class_positive_mask(metadata.labels)
        valid = valid_pair_mask(embeddings.shape[0], embeddings.device)
        negatives = valid & ~positives
        negative_lse = torch.logsumexp(
            similarities.masked_fill(~negatives, -torch.inf),
            dim=1,
        )
        pair_losses = torch.logaddexp(similarities, negative_lse[:, None]) - similarities
        positive_counts = positives.sum(dim=1)
        loss = _mean_normalized_anchor_loss(pair_losses, positives, positive_counts)
        return ObjectiveResult(
            loss,
            {
                "loss/sincere": loss.detach(),
                "pairs/positive_per_anchor": positive_counts.float().mean().detach(),
            },
        )


class SigmoidSupConObjective(Objective):
    def __init__(self, scale_init: float, bias_init: float | str) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale_init), dtype=torch.float32))
        initial_bias = 0.0 if bias_init == "auto" else float(bias_init)
        self.bias = nn.Parameter(torch.tensor(initial_bias, dtype=torch.float32))
        self.auto_bias = bias_init == "auto"
        self.register_buffer("_bias_initialized", torch.tensor(not self.auto_bias))

    def _initialize_bias(self, positives: torch.Tensor, valid: torch.Tensor) -> None:
        if bool(self._bias_initialized):
            return
        with torch.no_grad():
            prior = positives.sum().float() / valid.sum().float()
            prior = prior.clamp(1e-6, 1.0 - 1e-6)
            self.bias.copy_(torch.logit(prior))
            self._bias_initialized.fill_(True)

    def forward(self, output: ModelOutput, metadata: ObjectiveMetadata) -> ObjectiveResult:
        embeddings = output.embeddings.float()
        valid = valid_pair_mask(embeddings.shape[0], embeddings.device)
        positives = class_positive_mask(metadata.labels)
        negatives = valid & ~positives
        self._initialize_bias(positives, valid)
        scale = self.log_scale.exp().clamp(max=100.0)
        logits = scale * (embeddings @ embeddings.T) + self.bias
        signs = torch.where(positives, 1.0, -1.0)
        pair_losses = F.softplus(-signs * logits)
        positive_loss = _masked_mean(pair_losses, positives)
        negative_loss = _masked_mean(pair_losses, negatives)
        positive_counts = positives.sum(dim=1)
        loss = _mean_normalized_anchor_loss(pair_losses, valid, positive_counts)
        return ObjectiveResult(
            loss,
            {
                "loss/sigmoid": loss.detach(),
                "loss/sigmoid_positive": positive_loss.detach(),
                "loss/sigmoid_negative": negative_loss.detach(),
                "pairs/positive_per_anchor": positive_counts.float().mean().detach(),
                "sigmoid/scale": scale.detach(),
                "sigmoid/bias": self.bias.detach(),
            },
        )


class JointObjective(Objective):
    def __init__(
        self,
        temperature: float,
        cross_entropy_weight: float,
        contrastive_weight: float,
    ) -> None:
        super().__init__()
        self.ce = CrossEntropyObjective()
        self.supcon = SoftmaxContrastiveObjective(temperature, "class")
        self.cross_entropy_weight = cross_entropy_weight
        self.contrastive_weight = contrastive_weight

    def forward(self, output: ModelOutput, metadata: ObjectiveMetadata) -> ObjectiveResult:
        ce = self.ce(output, metadata)
        supcon = self.supcon(output, metadata)
        loss = self.cross_entropy_weight * ce.loss + self.contrastive_weight * supcon.loss
        return ObjectiveResult(
            loss,
            {
                **ce.metrics,
                **supcon.metrics,
                "loss/joint": loss.detach(),
            },
        )
