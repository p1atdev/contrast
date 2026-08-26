import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from contrast.config.schema import EvaluationConfig, LinearProbeConfig, PrecisionConfig
from contrast.models.model import ModelOutput
from contrast.runtime.precision import PrecisionManager
from contrast.training.evaluation import (
    EncodedDataset,
    _knn_accuracy,
    _linear_probe_accuracies,
    evaluate,
)


def _encoded(features: torch.Tensor, labels: torch.Tensor) -> EncodedDataset:
    return EncodedDataset(
        features=features,
        embeddings=F.normalize(features, dim=1),
        logits=torch.zeros(features.shape[0], 2),
        labels=labels,
    )


def test_knn_normalizes_backbone_features() -> None:
    memory = torch.tensor([[100.0, 0.0], [0.0, 1.0]])
    queries = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    labels = torch.tensor([0, 1])

    accuracy = _knn_accuracy(
        memory,
        labels,
        queries,
        labels,
        torch.device("cpu"),
        knn_k=1,
    )

    assert accuracy == 1.0


def test_linear_probe_is_deterministic_and_learns_frozen_features() -> None:
    memory = _encoded(
        torch.tensor(
            [
                [-2.0, -0.2],
                [-1.5, 0.1],
                [-1.0, -0.1],
                [-0.7, 0.2],
                [0.7, -0.2],
                [1.0, 0.1],
                [1.5, -0.1],
                [2.0, 0.2],
            ]
        ),
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    )
    query = _encoded(
        torch.tensor([[-1.2, 0.0], [-0.8, 0.1], [0.8, -0.1], [1.2, 0.0]]),
        torch.tensor([0, 0, 1, 1]),
    )
    config = LinearProbeConfig(
        epochs=30,
        batch_size=4,
        lr=0.2,
        momentum=0.0,
        seed=7,
    )

    first, first_loss = _linear_probe_accuracies(
        memory,
        {"eval": query},
        torch.device("cpu"),
        config,
    )
    second, second_loss = _linear_probe_accuracies(
        memory,
        {"eval": query},
        torch.device("cpu"),
        config,
    )

    assert first == {"eval": 1.0}
    assert second == first
    assert math.isfinite(first_loss)
    assert second_loss == first_loss


class SignModel(nn.Module):
    def forward(self, images: torch.Tensor) -> ModelOutput:
        features = images.float()
        return ModelOutput(
            features=features,
            embeddings=F.normalize(features, dim=1),
            logits=torch.stack((-features[:, 0], features[:, 0]), dim=1),
        )


def test_evaluate_reports_both_spaces_and_restores_model_mode() -> None:
    memory_features = torch.tensor([[-2.0, 0.0], [-1.0, 0.1], [1.0, -0.1], [2.0, 0.0]])
    memory_labels = torch.tensor([0, 0, 1, 1])
    query_features = torch.tensor([[-1.5, 0.0], [1.5, 0.0]])
    query_labels = torch.tensor([0, 1])
    memory_loader = DataLoader(
        TensorDataset(
            memory_features[:, None, :],
            memory_labels,
            torch.arange(memory_labels.numel()),
        ),
        batch_size=2,
    )
    query_loader = DataLoader(
        TensorDataset(
            query_features[:, None, :],
            query_labels,
            torch.arange(query_labels.numel()),
        ),
        batch_size=2,
    )
    model = SignModel()
    model.train()
    precision = PrecisionManager(PrecisionConfig(autocast_dtype="none"), torch.device("cpu"))
    config = EvaluationConfig(
        knn_k=1,
        linear_probe=LinearProbeConfig(enabled=False),
    )

    metrics = evaluate(
        model,
        memory_loader,
        {"eval": query_loader},
        torch.device("cpu"),
        precision,
        config,
    )

    assert metrics == {
        "eval/joint_classifier_top1": 1.0,
        "eval/backbone_knn_top1": 1.0,
        "eval/projector_knn_top1": 1.0,
    }
    assert model.training
