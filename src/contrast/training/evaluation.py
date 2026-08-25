from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from contrast.runtime.precision import PrecisionManager


@torch.no_grad()
def _encode(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: PrecisionManager,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = []
    logits = []
    labels = []
    for views, batch_labels, _ in loader:
        images = views[:, 0].to(device, non_blocking=True)
        with precision.autocast():
            output = model(images)
        embeddings.append(output.embeddings.float().cpu())
        logits.append(output.logits.float().cpu())
        labels.append(batch_labels)
    return torch.cat(embeddings), torch.cat(logits), torch.cat(labels)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    memory_loader: DataLoader,
    query_loader: DataLoader,
    device: torch.device,
    precision: PrecisionManager,
    knn_k: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    memory_embeddings, _, memory_labels = _encode(model, memory_loader, device, precision)
    query_embeddings, query_logits, query_labels = _encode(model, query_loader, device, precision)
    classifier_accuracy = (query_logits.argmax(1) == query_labels).float().mean()

    memory_embeddings = memory_embeddings.to(device)
    memory_labels = memory_labels.to(device)
    class_count = int(max(memory_labels.max(), query_labels.max()).item()) + 1
    correct = 0
    total = 0
    k = min(knn_k, memory_embeddings.shape[0])
    for query, labels in zip(query_embeddings.split(512), query_labels.split(512), strict=True):
        similarities = query.to(device) @ memory_embeddings.T
        values, indices = similarities.topk(k, dim=1)
        neighbor_labels = memory_labels[indices]
        votes = torch.zeros(query.shape[0], class_count, device=device)
        votes.scatter_add_(1, neighbor_labels, (values / 0.07).exp())
        predictions = votes.argmax(1).cpu()
        correct += int((predictions == labels).sum())
        total += labels.numel()
    model.train(was_training)
    return {
        "eval/classifier_top1": float(classifier_accuracy),
        "eval/knn_top1": correct / total,
    }
