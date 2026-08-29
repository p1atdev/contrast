from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from contrast.config.schema import PrecisionConfig
from contrast.models.model import ModelOutput
from contrast.objectives.base import ObjectiveMetadata
from contrast.objectives.self_supervised import (
    BarlowTwinsObjective,
    BYOLObjective,
    MoCoObjective,
)
from contrast.runtime.precision import PrecisionManager
from contrast.training.steps import DirectStep, GradCacheStep, PreparedBatch


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 6)
        self.projector = nn.Sequential(nn.Linear(6, 9), nn.GELU(), nn.Linear(9, 5))
        self.classifier = nn.Linear(6, 3)

    def forward(self, images: torch.Tensor) -> ModelOutput:
        features = torch.tanh(self.encoder(images))
        raw_embeddings = self.projector(features)
        return ModelOutput(
            features=features,
            embeddings=F.normalize(raw_embeddings, dim=-1),
            logits=self.classifier(features),
            raw_embeddings=raw_embeddings,
        )


def _batch() -> PreparedBatch:
    torch.manual_seed(23)
    return PreparedBatch(
        images=torch.randn(6, 4),
        metadata=ObjectiveMetadata(
            labels=torch.tensor([0, 0, 1, 1, 2, 2]),
            source_ids=torch.tensor([0, 0, 1, 1, 2, 2]),
            view_ids=torch.tensor([0, 1, 0, 1, 0, 1]),
        ),
    )


def _precision() -> PrecisionManager:
    return PrecisionManager(PrecisionConfig(autocast_dtype="none"), torch.device("cpu"))


def _gradients(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


def test_barlow_twins_uses_raw_projector_outputs() -> None:
    model = TinyModel()
    output = model(_batch().images)
    output.embeddings = output.embeddings.detach()
    objective = BarlowTwinsObjective(redundancy_weight=0.005, eps=1e-4)

    result = objective(output, _batch().metadata)

    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert output.raw_embeddings is not None
    assert output.raw_embeddings.grad_fn is not None
    assert model.projector[0].weight.grad is not None


def test_barlow_twins_requires_exactly_two_views() -> None:
    model = TinyModel()
    batch = _batch()
    output = model(batch.images)
    metadata = ObjectiveMetadata(
        labels=batch.metadata.labels,
        source_ids=batch.metadata.source_ids,
        view_ids=torch.tensor([0, 1, 2, 0, 1, 2]),
    )

    with pytest.raises(ValueError, match="exactly two"):
        BarlowTwinsObjective(0.005, 1e-4)(output, metadata)


def test_byol_updates_detached_target_after_optimizer_step() -> None:
    model = TinyModel()
    objective = BYOLObjective(
        embedding_dim=5,
        predictor_hidden_dim=7,
        base_target_decay=0.9,
        final_target_decay=1.0,
    )
    objective.initialize(model, total_steps=10)
    assert objective.target_model is not None
    original_target = deepcopy(objective.target_model.state_dict())

    result = DirectStep(_precision()).backward(model, objective, _batch())
    assert torch.isfinite(result.loss)
    assert all(parameter.grad is None for parameter in objective.target_model.parameters())
    assert objective.predictor[0].weight.grad is not None

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1)
    metrics = objective.after_optimizer_step(model, optimizer_step=1)

    assert int(objective.target_updates) == 1
    assert 0.9 < float(objective.last_target_decay) < 1.0
    assert metrics["target/updates"] == 1
    assert any(
        not torch.equal(original_target[name], value)
        for name, value in objective.target_model.state_dict().items()
    )
    objective.train()
    assert not objective.target_model.training


def test_moco_updates_queue_once_per_logical_batch_and_round_trips_state() -> None:
    model = TinyModel()
    objective = MoCoObjective(
        embedding_dim=5,
        queue_size=5,
        temperature=0.2,
        target_decay=0.95,
        symmetric=False,
    )
    objective.initialize(model, total_steps=10)
    initial_queue = objective.queue.clone()

    first = DirectStep(_precision()).backward(model, objective, _batch())
    assert torch.isfinite(first.loss)
    objective.after_optimizer_step(model, optimizer_step=1)
    assert int(objective.queue_pointer) == 3
    assert not torch.equal(objective.queue, initial_queue)

    model.zero_grad(set_to_none=True)
    second = DirectStep(_precision()).backward(model, objective, _batch())
    assert torch.isfinite(second.loss)
    objective.after_optimizer_step(model, optimizer_step=2)
    assert int(objective.queue_pointer) == 1

    restored = MoCoObjective(5, 5, 0.2, 0.95, False)
    restored.initialize(deepcopy(model), total_steps=10)
    restored.load_state_dict(objective.state_dict())
    torch.testing.assert_close(restored.queue, objective.queue)
    assert int(restored.queue_pointer) == int(objective.queue_pointer)
    assert int(restored.target_updates) == 2


@pytest.mark.parametrize(
    "objective_factory",
    [
        lambda: BarlowTwinsObjective(0.005, 1e-4),
        lambda: BYOLObjective(5, 7, 0.9, 1.0),
        lambda: MoCoObjective(5, 11, 0.2, 0.95, True),
    ],
)
def test_grad_cache_matches_direct_for_self_supervised_objectives(objective_factory) -> None:
    torch.manual_seed(31)
    direct_model = TinyModel()
    cache_model = deepcopy(direct_model)
    direct_objective = objective_factory()
    cache_objective = deepcopy(direct_objective)
    direct_objective.initialize(direct_model, total_steps=10)
    cache_objective.initialize(cache_model, total_steps=10)
    cache_objective.load_state_dict(direct_objective.state_dict())
    batch = _batch()

    direct = DirectStep(_precision()).backward(direct_model, direct_objective, batch)
    cached = GradCacheStep(_precision(), chunk_size=2).backward(
        cache_model,
        cache_objective,
        batch,
    )

    torch.testing.assert_close(cached.loss, direct.loss)
    direct_gradients = {
        **{f"model.{key}": value for key, value in _gradients(direct_model).items()},
        **{f"objective.{key}": value for key, value in _gradients(direct_objective).items()},
    }
    cache_gradients = {
        **{f"model.{key}": value for key, value in _gradients(cache_model).items()},
        **{f"objective.{key}": value for key, value in _gradients(cache_objective).items()},
    }
    assert cache_gradients.keys() == direct_gradients.keys()
    for name in direct_gradients:
        torch.testing.assert_close(
            cache_gradients[name],
            direct_gradients[name],
            atol=2e-5,
            rtol=1e-4,
        )

    if isinstance(direct_objective, MoCoObjective):
        direct_objective.after_optimizer_step(direct_model, 1)
        cache_objective.after_optimizer_step(cache_model, 1)
        torch.testing.assert_close(cache_objective.queue, direct_objective.queue)
