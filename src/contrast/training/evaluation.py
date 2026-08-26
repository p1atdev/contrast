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
