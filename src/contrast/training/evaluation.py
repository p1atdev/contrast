from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from contrast.config.schema import EvaluationConfig, LinearProbeConfig
from contrast.runtime.precision import PrecisionManager


@dataclass(frozen=True)
class EncodedDataset:
    features: torch.Tensor
    embeddings: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor


@torch.no_grad()
def _encode(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: PrecisionManager,
) -> EncodedDataset:
    features = []
    embeddings = []
    logits = []
    labels = []
    for views, batch_labels, _ in loader:
        images = views[:, 0].to(device, non_blocking=True)
        with precision.autocast():
            output = model(images)
        features.append(output.features.float().cpu())
        embeddings.append(output.embeddings.float().cpu())
        logits.append(output.logits.float().cpu())
        labels.append(batch_labels)
    return EncodedDataset(
        features=torch.cat(features),
        embeddings=torch.cat(embeddings),
        logits=torch.cat(logits),
        labels=torch.cat(labels),
    )


@torch.no_grad()
def _knn_accuracy(
    memory_vectors: torch.Tensor,
    memory_labels: torch.Tensor,
    query_vectors: torch.Tensor,
    query_labels: torch.Tensor,
    device: torch.device,
    knn_k: int,
) -> float:
    memory_vectors = F.normalize(memory_vectors.float(), dim=1).to(device)
    memory_labels = memory_labels.to(device)
    class_count = int(max(memory_labels.max(), query_labels.max()).item()) + 1
    correct = 0
    total = 0
    k = min(knn_k, memory_vectors.shape[0])
    query_vectors = F.normalize(query_vectors.float(), dim=1)
    for query, labels in zip(query_vectors.split(512), query_labels.split(512), strict=True):
        similarities = query.to(device) @ memory_vectors.T
        values, indices = similarities.topk(k, dim=1)
        neighbor_labels = memory_labels[indices]
        votes = torch.zeros(query.shape[0], class_count, device=device)
        votes.scatter_add_(1, neighbor_labels, (values / 0.07).exp())
        predictions = votes.argmax(1).cpu()
        correct += int((predictions == labels).sum())
        total += labels.numel()
    return correct / total


@torch.no_grad()
def _representation_diagnostics(vectors: torch.Tensor) -> dict[str, float]:
    """Scale-free diagnostics for collapse and dimensional redundancy.

    Vectors are L2-normalized before measurement so backbone and projector
    spaces remain comparable. ``isotropy`` is the mean per-dimension standard
    deviation relative to the unit-sphere maximum ``1 / sqrt(d)``. Effective
    rank is the entropy rank of the centered covariance, divided by ``d``.
    """
    if vectors.ndim != 2 or vectors.shape[0] < 2 or vectors.shape[1] < 1:
        raise ValueError("representation diagnostics require a [N, D] tensor with N >= 2")

    normalized = F.normalize(vectors.float(), dim=1)
    sample_count, dimension = normalized.shape
    centered = normalized - normalized.mean(dim=0)
    covariance = centered.T @ centered / sample_count
    variances = covariance.diagonal().clamp_min(0.0)
    standard_deviations = variances.sqrt()
    std_mean = standard_deviations.mean()
    isotropy = (std_mean * math.sqrt(dimension)).clamp(0.0, 1.0)

    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total_variance = eigenvalues.sum()
    if float(total_variance) > torch.finfo(eigenvalues.dtype).eps:
        probabilities = eigenvalues / total_variance
        positive = probabilities > 0
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
        effective_rank_ratio = (entropy.exp() / dimension).clamp(0.0, 1.0)
    else:
        effective_rank_ratio = eigenvalues.new_zeros(())

    scales = standard_deviations[:, None] * standard_deviations[None, :]
    off_diagonal = ~torch.eye(dimension, dtype=torch.bool, device=vectors.device)
    valid_correlations = off_diagonal & (scales > torch.finfo(scales.dtype).eps)
    if bool(valid_correlations.any()):
        correlations = covariance / scales.clamp_min(torch.finfo(scales.dtype).eps)
        offdiag_correlation_rms = correlations[valid_correlations].square().mean().sqrt()
    else:
        offdiag_correlation_rms = covariance.new_zeros(())

    summed = normalized.sum(dim=0)
    pairwise_cosine_sum = summed.square().sum() - normalized.square().sum()
    mean_pairwise_cosine = (pairwise_cosine_sum / (sample_count * (sample_count - 1))).clamp(
        -1.0, 1.0
    )

    return {
        "std_mean": float(std_mean),
        "isotropy": float(isotropy),
        "effective_rank_ratio": float(effective_rank_ratio),
        "offdiag_correlation_rms": float(offdiag_correlation_rms),
        "mean_pairwise_cosine": float(mean_pairwise_cosine),
    }


def _linear_probe_accuracies(
    memory: EncodedDataset,
    queries: Mapping[str, EncodedDataset],
    device: torch.device,
    config: LinearProbeConfig,
) -> tuple[dict[str, float], float]:
    feature_dim = memory.features.shape[1]
    class_count = (
        max(
            int(memory.labels.max()),
            *(int(query.labels.max()) for query in queries.values()),
        )
        + 1
    )
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(config.seed)
        classifier = nn.Linear(feature_dim, class_count).to(device)
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        generator = torch.Generator().manual_seed(config.seed)
        batches_per_epoch = math.ceil(memory.features.shape[0] / config.batch_size)
        total_steps = config.epochs * batches_per_epoch
        step = 0
        final_loss = float("nan")
        classifier.train()
        for _ in range(config.epochs):
            order = torch.randperm(memory.features.shape[0], generator=generator)
            for indices in order.split(config.batch_size):
                progress = step / max(total_steps - 1, 1)
                lr = config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                features = memory.features[indices].to(device, non_blocking=True)
                labels = memory.labels[indices].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(classifier(features), labels)
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach())
                step += 1

        classifier.eval()
        accuracies: dict[str, float] = {}
        with torch.no_grad():
            for name, query in queries.items():
                correct = 0
                total = 0
                feature_batches = query.features.split(config.batch_size)
                label_batches = query.labels.split(config.batch_size)
                for features, labels in zip(feature_batches, label_batches, strict=True):
                    predictions = classifier(features.to(device, non_blocking=True)).argmax(1).cpu()
                    correct += int((predictions == labels).sum())
                    total += labels.numel()
                accuracies[name] = correct / total
    return accuracies, final_loss


def evaluate(
    model: nn.Module,
    memory_loader: DataLoader,
    query_loaders: Mapping[str, DataLoader],
    device: torch.device,
    precision: PrecisionManager,
    config: EvaluationConfig,
    *,
    include_linear_probe: bool = False,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        memory = _encode(model, memory_loader, device, precision)
        queries = {
            name: _encode(model, loader, device, precision)
            for name, loader in query_loaders.items()
        }
        metrics: dict[str, float] = {}
        for name, query in queries.items():
            metrics[f"{name}/joint_classifier_top1"] = float(
                (query.logits.argmax(1) == query.labels).float().mean()
            )
            for space in config.knn_spaces:
                memory_vectors = memory.features if space == "backbone" else memory.embeddings
                query_vectors = query.features if space == "backbone" else query.embeddings
                metrics[f"{name}/{space}_knn_top1"] = _knn_accuracy(
                    memory_vectors,
                    memory.labels,
                    query_vectors,
                    query.labels,
                    device,
                    config.knn_k,
                )
                metrics.update(
                    {
                        f"{name}/{space}_{metric}": value
                        for metric, value in _representation_diagnostics(query_vectors).items()
                    }
                )
        if include_linear_probe and config.linear_probe.enabled:
            accuracies, final_loss = _linear_probe_accuracies(
                memory,
                queries,
                device,
                config.linear_probe,
            )
            primary_query = next(iter(queries))
            metrics[f"{primary_query}/linear_probe_train_loss"] = final_loss
            metrics.update(
                {f"{name}/linear_probe_top1": accuracy for name, accuracy in accuracies.items()}
            )
        return metrics
    finally:
        model.train(was_training)
