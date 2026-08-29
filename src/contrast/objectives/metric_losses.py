"""Supervised metric-learning objectives.

The objectives in this module deliberately keep their own learnable class
proxies.  This makes a proxy objective self contained (and, importantly,
means that its proxies are included in ``objective.parameters()``) while
leaving the model's classifier head untouched.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from contrast.models.model import ModelOutput
from contrast.objectives.base import Objective, ObjectiveMetadata, ObjectiveResult


def _validate_batch(output: ModelOutput, metadata: ObjectiveMetadata) -> None:
    batch_size = output.features.shape[0]
    if batch_size == 0:
        raise ValueError("metric objectives require a non-empty batch")
    if output.features.ndim != 2 or output.embeddings.ndim != 2:
        raise ValueError("metric objectives expect two-dimensional features and embeddings")
    if output.embeddings.shape[0] != batch_size:
        raise ValueError("features and embeddings must have the same batch size")
    for name, value in (
        ("labels", metadata.labels),
        ("source_ids", metadata.source_ids),
        ("view_ids", metadata.view_ids),
    ):
        if value.ndim != 1 or value.numel() != batch_size:
            raise ValueError(f"{name} must be a one-dimensional tensor of batch size")


def _require_two_views(metadata: ObjectiveMetadata) -> None:
    """Ensure that a contrastive/metric batch contains the expected two views.

    We intentionally check distinct view ids rather than assuming that views
    are ordered or that the batch size is divisible by two.  This catches
    accidentally flattened single-view batches while still allowing a final
    uneven batch.
    """

    if torch.unique(metadata.view_ids).numel() < 2:
        raise ValueError("metric objectives require at least two views")


def _normalized_embeddings(output: ModelOutput) -> torch.Tensor:
    return F.normalize(output.embeddings.float(), dim=-1, eps=1e-12)


def _pair_masks(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = ~torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None, :]) & valid
    negatives = valid & ~positives
    return positives, negatives


def _safe_logsumexp(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    counts = mask.sum(dim=1)
    lse = torch.logsumexp(values.masked_fill(~mask, -torch.inf), dim=1)
    # logsumexp(empty) is -inf.  Zero is the neutral value for the optional
    # positive/negative branch and keeps the returned loss finite.
    lse = torch.where(counts > 0, lse, torch.zeros_like(lse))
    return lse, counts


def _optional_softplus(values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return torch.where(counts > 0, F.softplus(values), torch.zeros_like(values))


def _anchor_mean(values: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    if bool(active.any()):
        return values[active].mean()
    # A batch with no valid pair/triplet is a legitimate final or distributed
    # batch.  Return a differentiable zero rather than NaN or an exception.
    return values.sum() * 0.0


def _metric_result(
    loss: torch.Tensor,
    name: str,
    *,
    positive_counts: torch.Tensor | None = None,
    negative_counts: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
) -> ObjectiveResult:
    metrics: dict[str, torch.Tensor] = {f"loss/{name}": loss.detach()}
    if positive_counts is not None:
        metrics["pairs/positive_per_anchor"] = positive_counts.float().mean().detach()
    if negative_counts is not None:
        metrics["pairs/negative_per_anchor"] = negative_counts.float().mean().detach()
    if active is not None:
        metrics["anchors/active"] = active.float().sum().detach()
        metrics["anchors/total"] = torch.tensor(float(active.numel()), device=loss.device)
    return ObjectiveResult(loss, metrics)


class _ProxyObjective(Objective):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        scale: float,
        *,
        proxies: torch.Tensor | None = None,
        feature_source: Literal["features", "embeddings"] = "features",
    ) -> None:
        super().__init__()
        if num_classes <= 0 or feature_dim <= 0:
            raise ValueError("num_classes and feature_dim must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")
        if feature_source not in {"features", "embeddings"}:
            raise ValueError("feature_source must be 'features' or 'embeddings'")
        if proxies is None:
            initial = torch.randn(num_classes, feature_dim) * math.sqrt(2.0 / feature_dim)
        else:
            if tuple(proxies.shape) != (num_classes, feature_dim):
                raise ValueError("proxies have the wrong shape")
            initial = proxies.detach().float().clone()
        self.proxies = nn.Parameter(initial)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.scale = float(scale)
        self.feature_source = feature_source

    @property
    def class_proxies(self) -> nn.Parameter:
        """Readable alias used by callers that distinguish class proxies."""

        return self.proxies

    def _features(self, output: ModelOutput) -> torch.Tensor:
        raw = output.features if self.feature_source == "features" else output.embeddings
        if raw.shape[1] != self.feature_dim:
            raise ValueError("objective feature_dim does not match ModelOutput")
        return F.normalize(raw.float(), dim=-1, eps=1e-12)

    def _cosine_logits(self, output: ModelOutput) -> torch.Tensor:
        return self._features(output) @ F.normalize(self.proxies.float(), dim=-1, eps=1e-12).T

    def _validate(self, output: ModelOutput, metadata: ObjectiveMetadata) -> None:
        _validate_batch(output, metadata)
        _require_two_views(metadata)
        if int(metadata.labels.min()) < 0 or int(metadata.labels.max()) >= self.num_classes:
            raise ValueError("labels must be in [0, num_classes)")


class NormalizedSoftmaxObjective(_ProxyObjective):
    """Cross entropy over cosine-normalized features and class proxies."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        scale: float = 32.0,
        *,
        proxies: torch.Tensor | None = None,
        feature_source: Literal["features", "embeddings"] = "features",
    ) -> None:
        super().__init__(
            num_classes, feature_dim, scale, proxies=proxies, feature_source=feature_source
        )

    def forward(self, output: ModelOutput, metadata: ObjectiveMetadata) -> ObjectiveResult:
        self._validate(output, metadata)
        logits = self.scale * self._cosine_logits(output)
        loss = F.cross_entropy(logits, metadata.labels)
        result = _metric_result(loss, "normalized_softmax")
        result.metrics.update(
            {
                "proxy/top1": logits.detach().argmax(dim=1).eq(metadata.labels).float().mean(),
                "proxy/scale": torch.tensor(self.scale, device=loss.device),
            }
        )
        return result


class CosFaceObjective(_ProxyObjective):
    """Additive cosine-margin softmax (CosFace)."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        scale: float = 32.0,
        margin: float = 0.35,
        *,
        proxies: torch.Tensor | None = None,
        feature_source: Literal["features", "embeddings"] = "features",
    ) -> None:
        super().__init__(
            num_classes, feature_dim, scale, proxies=proxies, feature_source=feature_source
        )
        if margin < 0 or margin >= 1:
            raise ValueError("CosFace margin must be in [0, 1)")
        self.margin = float(margin)

    def forward(self, output: ModelOutput, metadata: ObjectiveMetadata) -> ObjectiveResult:
        self._validate(output, metadata)
        cosine = self._cosine_logits(output)
        target = cosine - (
            F.one_hot(metadata.labels, self.num_classes).to(cosine.dtype) * self.margin
        )
        logits = self.scale * target
        loss = F.cross_entropy(logits, metadata.labels)
        result = _metric_result(loss, "cosface")
        result.metrics.update(
            {
                "proxy/top1": logits.detach().argmax(dim=1).eq(metadata.labels).float().mean(),
                "proxy/scale": torch.tensor(self.scale, device=loss.device),
                "proxy/margin": torch.tensor(self.margin, device=loss.device),
            }
        )
        return result


class ArcFaceObjective(_ProxyObjective):
    """Additive angular-margin softmax (ArcFace)."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        scale: float = 32.0,
        margin: float = 0.5,
        *,
        proxies: torch.Tensor | None = None,
        feature_source: Literal["features", "embeddings"] = "features",
    ) -> None:
        super().__init__(
            num_classes, feature_dim, scale, proxies=proxies, feature_source=feature_source
        )
        if margin < 0 or margin >= math.pi:
            raise ValueError("ArcFace margin must be in [0, pi)")
        self.margin = float(margin)

    def forward(self, output: ModelOutput, metadata: ObjectiveMetadata) -> ObjectiveResult:
        self._validate(output, metadata)
        cosine = self._cosine_logits(output)
        eps = torch.finfo(cosine.dtype).eps
        safe_cosine = cosine.clamp(-1.0 + eps, 1.0 - eps)
        sine = torch.sqrt((1.0 - safe_cosine.square()).clamp_min(0.0))
        cosine_m = math.cos(self.margin)
        sine_m = math.sin(self.margin)
        target = safe_cosine * cosine_m - sine * sine_m
        threshold = math.cos(math.pi - self.margin)
        correction = math.sin(math.pi - self.margin) * self.margin
        target = torch.where(safe_cosine > threshold, target, safe_cosine - correction)
        target_mask = F.one_hot(metadata.labels, self.num_classes).to(cosine.dtype)
        logits = self.scale * (cosine * (1.0 - target_mask) + target * target_mask)
        loss = F.cross_entropy(logits, metadata.labels)
        result = _metric_result(loss, "arcface")
        result.metrics.update(
            {
                "proxy/top1": logits.detach().argmax(dim=1).eq(metadata.labels).float().mean(),
                "proxy/scale": torch.tensor(self.scale, device=loss.device),
                "proxy/margin": torch.tensor(self.margin, device=loss.device),
            }
        )
        return result


class CircleLossObjective(Objective):
    """Circle loss over normalized embeddings and supervised pairs."""

    def __init__(self, scale: float = 80.0, margin: float = 0.25) -> None:
        super().__init__()
        if scale <= 0 or margin < 0 or margin >= 1:
            raise ValueError("CircleLoss requires scale > 0 and margin in [0, 1)")
        self.scale = float(scale)
        self.margin = float(margin)

    def forward(self, output: ModelOutput, metadata: ObjectiveMetadata) -> ObjectiveResult:
        _validate_batch(output, metadata)
        _require_two_views(metadata)
        embeddings = _normalized_embeddings(output)
        positives, negatives = _pair_masks(metadata.labels)
        similarities = embeddings @ embeddings.T
        alpha_p = F.relu(-similarities + 1.0 + self.margin).detach()
        alpha_n = F.relu(similarities + self.margin).detach()
        delta_p = 1.0 - self.margin
        delta_n = self.margin
        pos_logits = -self.scale * alpha_p * (similarities - delta_p)
        neg_logits = self.scale * alpha_n * (similarities - delta_n)
        pos_lse, pos_counts = _safe_logsumexp(pos_logits, positives)
        neg_lse, neg_counts = _safe_logsumexp(neg_logits, negatives)
        per_anchor = F.softplus(pos_lse + neg_lse)
        active = (pos_counts > 0) & (neg_counts > 0)
        loss = _anchor_mean(per_anchor, active)
        return _metric_result(
            loss,
            "circle",
            positive_counts=pos_counts,
            negative_counts=neg_counts,
            active=active,
        )


class ProxyAnchorObjective(_ProxyObjective):
    """Proxy Anchor loss with one learnable proxy per class."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        scale: float = 32.0,
        margin: float = 0.1,
        *,
        proxies: torch.Tensor | None = None,
        feature_source: Literal["features", "embeddings"] = "embeddings",
    ) -> None:
        super().__init__(
            num_classes, feature_dim, scale, proxies=proxies, feature_source=feature_source
        )
        if margin < 0:
            raise ValueError("ProxyAnchor margin must be non-negative")
        self.margin = float(margin)

    def forward(self, output: ModelOutput, metadata: ObjectiveMetadata) -> ObjectiveResult:
        self._validate(output, metadata)
        embeddings = self._features(output)
        proxies = F.normalize(self.proxies.float(), dim=-1, eps=1e-12)
        similarities = embeddings @ proxies.T
        class_ids = torch.arange(self.num_classes, device=similarities.device)
        positive = metadata.labels[:, None].eq(class_ids[None, :])
        negative = ~positive
        pos_logits = -self.scale * (similarities - self.margin)
        neg_logits = self.scale * (similarities + self.margin)
        pos_lse, pos_counts = _safe_logsumexp(pos_logits.T, positive.T)
        neg_lse, neg_counts = _safe_logsumexp(neg_logits.T, negative.T)
        pos_active = pos_counts > 0
        neg_active = neg_counts > 0
        pos_term = (
            F.softplus(pos_lse[pos_active]).mean()
            if bool(pos_active.any())
            else similarities.sum() * 0.0
        )
        neg_term = (
            F.softplus(neg_lse[neg_active]).mean()
            if bool(neg_active.any())
            else similarities.sum() * 0.0
        )
        loss = pos_term + neg_term
        result = _metric_result(loss, "proxy_anchor")
        result.metrics.update(
            {
                "proxy/top1": (
                    similarities.detach().argmax(dim=1).eq(metadata.labels).float().mean()
                ),
                "proxy/active_positive": pos_active.float().sum().detach(),
                "proxy/active_negative": neg_active.float().sum().detach(),
                "proxy/scale": torch.tensor(self.scale, device=loss.device),
                "proxy/margin": torch.tensor(self.margin, device=loss.device),
            }
        )
        return result


class BatchHardTripletObjective(Objective):
    """Batch-hard triplet loss using cosine distance on normalized embeddings."""

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        if margin < 0:
            raise ValueError("triplet margin must be non-negative")
        self.margin = float(margin)

    def forward(self, output: ModelOutput, metadata: ObjectiveMetadata) -> ObjectiveResult:
        _validate_batch(output, metadata)
        _require_two_views(metadata)
        embeddings = _normalized_embeddings(output)
        positives, negatives = _pair_masks(metadata.labels)
        similarities = embeddings @ embeddings.T
        pos_counts = positives.sum(dim=1)
        neg_counts = negatives.sum(dim=1)
        hard_positive = similarities.masked_fill(~positives, torch.inf).min(dim=1).values
        hard_negative = similarities.masked_fill(~negatives, -torch.inf).max(dim=1).values
        active = (pos_counts > 0) & (neg_counts > 0)
        per_anchor = F.relu(hard_negative - hard_positive + self.margin)
        per_anchor = torch.where(active, per_anchor, torch.zeros_like(per_anchor))
        loss = _anchor_mean(per_anchor, active)
        return _metric_result(
            loss,
            "batch_hard_triplet",
            positive_counts=pos_counts,
            negative_counts=neg_counts,
            active=active,
        )


class MultiSimilarityObjective(Objective):
    """Multi-Similarity loss with standard easy-pair mining."""

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 50.0,
        base: float = 0.5,
        margin: float = 0.1,
    ) -> None:
        super().__init__()
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        if margin < 0:
            raise ValueError("mining margin must be non-negative")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.base = float(base)
        self.margin = float(margin)

    def forward(self, output: ModelOutput, metadata: ObjectiveMetadata) -> ObjectiveResult:
        _validate_batch(output, metadata)
        _require_two_views(metadata)
        embeddings = _normalized_embeddings(output)
        positives, negatives = _pair_masks(metadata.labels)
        similarities = embeddings @ embeddings.T

        # Standard MS mining keeps positives below the hardest negative and
        # negatives above the easiest positive.  The fallbacks preserve all
        # available pairs when one side is absent for an anchor.
        neg_max = similarities.masked_fill(~negatives, -torch.inf).max(dim=1).values
        pos_min = similarities.masked_fill(~positives, torch.inf).min(dim=1).values
        mined_positive = positives & (similarities < neg_max[:, None] + self.margin)
        mined_negative = negatives & (similarities > pos_min[:, None] - self.margin)
        mined_positive = torch.where(torch.isfinite(neg_max[:, None]), mined_positive, positives)
        mined_negative = torch.where(torch.isfinite(pos_min[:, None]), mined_negative, negatives)

        pos_values = -self.alpha * (similarities - self.base)
        neg_values = self.beta * (similarities - self.base)
        pos_lse, pos_counts = _safe_logsumexp(pos_values, mined_positive)
        neg_lse, neg_counts = _safe_logsumexp(neg_values, mined_negative)
        per_anchor = _optional_softplus(pos_lse, pos_counts) / self.alpha
        per_anchor += _optional_softplus(neg_lse, neg_counts) / self.beta
        active = (pos_counts > 0) & (neg_counts > 0)
        loss = _anchor_mean(per_anchor, active)
        return _metric_result(
            loss,
            "multi_similarity",
            positive_counts=pos_counts,
            negative_counts=neg_counts,
            active=active,
        )
