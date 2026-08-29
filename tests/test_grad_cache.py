from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from contrast.config.schema import PrecisionConfig
from contrast.models.model import ModelOutput
from contrast.objectives.base import Objective, ObjectiveMetadata, ObjectiveResult
from contrast.objectives.losses import SoftmaxContrastiveObjective
from contrast.runtime.precision import PrecisionManager
from contrast.training.steps import DirectStep, GradCacheStep, PreparedBatch


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 7)
        self.projector = nn.Linear(7, 5)
        self.classifier = nn.Linear(7, 3)

    def forward(self, images: torch.Tensor) -> ModelOutput:
        features = torch.tanh(self.encoder(images))
        return ModelOutput(
            features=features,
            embeddings=F.normalize(self.projector(features), dim=-1),
            logits=self.classifier(features),
        )


def gradients(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def test_grad_cache_matches_direct_gradients() -> None:
    torch.manual_seed(9)
    direct_model = TinyModel()
    cache_model = deepcopy(direct_model)
    images = torch.randn(8, 4)
    metadata = ObjectiveMetadata(
        labels=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        source_ids=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        view_ids=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
    )
    batch = PreparedBatch(images, metadata)
    precision = PrecisionManager(PrecisionConfig(autocast_dtype="none"), torch.device("cpu"))

    direct_result = DirectStep(precision).backward(
        direct_model,
        SoftmaxContrastiveObjective(0.2, "class"),
        batch,
    )
    cache_result = GradCacheStep(precision, chunk_size=3).backward(
        cache_model,
        SoftmaxContrastiveObjective(0.2, "class"),
        batch,
    )

    torch.testing.assert_close(cache_result.loss, direct_result.loss)
    direct_gradients = gradients(direct_model)
    cache_gradients = gradients(cache_model)
    assert direct_gradients.keys() == cache_gradients.keys()
    for name in direct_gradients:
        torch.testing.assert_close(
            cache_gradients[name], direct_gradients[name], rtol=1e-5, atol=1e-6
        )


class NonFiniteObjective(Objective):
    def forward(
        self,
        output: ModelOutput,
        metadata: ObjectiveMetadata,
    ) -> ObjectiveResult:
        del metadata
        loss = output.embeddings.sum() * output.embeddings.new_tensor(float("nan"))
        return ObjectiveResult(loss, {})


def test_training_steps_reject_non_finite_loss() -> None:
    precision = PrecisionManager(PrecisionConfig(autocast_dtype="none"), torch.device("cpu"))
    images = torch.randn(4, 4)
    metadata = ObjectiveMetadata(
        labels=torch.tensor([0, 0, 1, 1]),
        source_ids=torch.tensor([0, 0, 1, 1]),
        view_ids=torch.tensor([0, 1, 0, 1]),
    )
    batch = PreparedBatch(images, metadata)

    for strategy in (DirectStep(precision), GradCacheStep(precision, chunk_size=2)):
        with pytest.raises(FloatingPointError, match="non-finite"):
            strategy.backward(TinyModel(), NonFiniteObjective(), batch)
