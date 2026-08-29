from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelOutput:
    features: torch.Tensor
    embeddings: torch.Tensor
    logits: torch.Tensor
    raw_embeddings: torch.Tensor | None = None

    def as_dict(self) -> dict[str, torch.Tensor]:
        values = {
            "features": self.features,
            "embeddings": self.embeddings,
            "logits": self.logits,
        }
        if self.raw_embeddings is not None:
            values["raw_embeddings"] = self.raw_embeddings
        return values


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        projection_hidden_dim: int,
        embedding_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(feature_dim, projection_hidden_dim, embedding_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, images: torch.Tensor) -> ModelOutput:
        features = self.encoder(images)
        raw_embeddings = self.projector(features)
        return ModelOutput(
            features=features,
            embeddings=F.normalize(raw_embeddings, dim=-1),
            logits=self.classifier(features),
            raw_embeddings=raw_embeddings,
        )
