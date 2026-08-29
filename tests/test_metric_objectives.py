import pytest
import torch

from contrast.models.model import ModelOutput
from contrast.objectives.base import ObjectiveMetadata
from contrast.objectives.metric_losses import (
    ArcFaceObjective,
    BatchHardTripletObjective,
    CircleLossObjective,
    CosFaceObjective,
    MultiSimilarityObjective,
    NormalizedSoftmaxObjective,
    ProxyAnchorObjective,
)


def _problem(
    labels: torch.Tensor | None = None,
) -> tuple[ModelOutput, ObjectiveMetadata]:
    torch.manual_seed(7)
    labels = labels if labels is not None else torch.tensor([0, 0, 1, 1, 2, 2])
    batch_size = labels.numel()
    output = ModelOutput(
        features=torch.randn(batch_size, 6, requires_grad=True),
        embeddings=torch.randn(batch_size, 5, requires_grad=True),
        logits=torch.randn(batch_size, 3, requires_grad=True),
    )
    metadata = ObjectiveMetadata(
        labels=labels,
        source_ids=torch.arange(batch_size) // 2,
        view_ids=torch.arange(batch_size) % 2,
    )
    return output, metadata


@pytest.mark.parametrize(
    "objective",
    [
        NormalizedSoftmaxObjective(3, 6),
        CosFaceObjective(3, 6),
        ArcFaceObjective(3, 6),
        CircleLossObjective(),
        ProxyAnchorObjective(3, 5),
        BatchHardTripletObjective(),
        MultiSimilarityObjective(),
    ],
)
def test_metric_objective_is_finite_and_differentiable(objective: torch.nn.Module) -> None:
    output, metadata = _problem()
    result = objective(output, metadata)
    assert torch.isfinite(result.loss)
    assert result.metrics
    result.loss.backward()
    assert output.features.grad is not None or output.embeddings.grad is not None
    if hasattr(objective, "proxies"):
        assert objective.proxies.grad is not None
        assert torch.isfinite(objective.proxies.grad).all()


@pytest.mark.parametrize(
    "objective",
    [
        NormalizedSoftmaxObjective(3, 6),
        CosFaceObjective(3, 6),
        ArcFaceObjective(3, 6),
        CircleLossObjective(),
        ProxyAnchorObjective(3, 5),
        BatchHardTripletObjective(),
        MultiSimilarityObjective(),
    ],
)
def test_metric_objectives_require_two_views(objective: torch.nn.Module) -> None:
    output, metadata = _problem()
    single_view = ObjectiveMetadata(
        labels=metadata.labels,
        source_ids=metadata.source_ids,
        view_ids=torch.zeros_like(metadata.view_ids),
    )
    with pytest.raises(ValueError, match="two views"):
        objective(output, single_view)


@pytest.mark.parametrize(
    "objective",
    [CircleLossObjective(), BatchHardTripletObjective(), MultiSimilarityObjective()],
)
def test_empty_positive_anchors_are_skipped(objective: torch.nn.Module) -> None:
    # Class 0 has a pair; classes 1 and 2 are singleton anchors.  Their empty
    # positive sets must not produce -inf/NaN or poison the batch mean.
    output, metadata = _problem(torch.tensor([0, 0, 1, 2]))
    result = objective(output, metadata)
    assert torch.isfinite(result.loss)
    # The singleton anchors still have valid negative pairs; only their
    # positive branch is empty and must be neutral, not NaN.
    assert result.metrics["pairs/positive_per_anchor"] == pytest.approx(0.5)
    result.loss.backward()
    assert output.embeddings.grad is not None


def test_arcface_is_stable_at_cosine_boundaries() -> None:
    features = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
        requires_grad=True,
    )
    output = ModelOutput(
        features=features,
        embeddings=torch.randn(4, 3, requires_grad=True),
        logits=torch.zeros(4, 2),
    )
    metadata = ObjectiveMetadata(
        labels=torch.tensor([0, 0, 1, 1]),
        source_ids=torch.tensor([0, 0, 1, 1]),
        view_ids=torch.tensor([0, 1, 0, 1]),
    )
    objective = ArcFaceObjective(
        2,
        2,
        scale=64.0,
        margin=0.5,
        proxies=torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
    )
    result = objective(output, metadata)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(objective.proxies.grad).all()


def test_proxy_constructor_keeps_proxy_as_trainable_parameter() -> None:
    objective = NormalizedSoftmaxObjective(
        num_classes=2,
        feature_dim=4,
        proxies=torch.ones(2, 4),
    )
    assert list(objective.parameters()) == [objective.proxies]
    assert objective.proxies.requires_grad


def test_arcface_zero_margin_matches_normalized_softmax() -> None:
    output, metadata = _problem()
    proxies = torch.randn(3, 6)
    normalized = NormalizedSoftmaxObjective(3, 6, scale=24.0, proxies=proxies)
    arcface = ArcFaceObjective(3, 6, scale=24.0, margin=0.0, proxies=proxies)

    normalized_result = normalized(output, metadata)
    arcface_result = arcface(output, metadata)

    torch.testing.assert_close(arcface_result.loss, normalized_result.loss)


@pytest.mark.parametrize(
    "objective",
    [
        CosFaceObjective(
            2,
            2,
            scale=32.0,
            margin=0.35,
            proxies=torch.tensor([[1.0, 0.0], [0.9, 0.4358899]]),
        ),
        ArcFaceObjective(
            2,
            2,
            scale=32.0,
            margin=0.5,
            proxies=torch.tensor([[1.0, 0.0], [0.9, 0.4358899]]),
        ),
    ],
)
def test_margin_proxy_top1_uses_unmodified_cosine(objective: torch.nn.Module) -> None:
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    output = ModelOutput(
        features=features,
        embeddings=torch.randn(2, 3),
        logits=torch.zeros(2, 2),
    )
    metadata = ObjectiveMetadata(
        labels=torch.zeros(2, dtype=torch.long),
        source_ids=torch.zeros(2, dtype=torch.long),
        view_ids=torch.arange(2),
    )

    result = objective(output, metadata)

    assert result.metrics["proxy/top1"] == 1.0
    assert result.metrics["proxy/margin_adjusted_top1"] == 0.0


def test_circle_loss_matches_pairwise_reference_formula() -> None:
    output, metadata = _problem()
    objective = CircleLossObjective(scale=32.0, margin=0.25)

    result = objective(output, metadata)

    embeddings = torch.nn.functional.normalize(output.embeddings.float(), dim=-1)
    similarities = embeddings @ embeddings.T
    valid = ~torch.eye(similarities.shape[0], dtype=torch.bool)
    positives = metadata.labels[:, None].eq(metadata.labels[None, :]) & valid
    negatives = valid & ~positives
    alpha_positive = torch.relu(-similarities + 1.25).detach()
    alpha_negative = torch.relu(similarities + 0.25).detach()
    positive_logits = -32.0 * alpha_positive * (similarities - 0.75)
    negative_logits = 32.0 * alpha_negative * (similarities - 0.25)
    expected = torch.nn.functional.softplus(
        torch.logsumexp(positive_logits.masked_fill(~positives, -torch.inf), dim=1)
        + torch.logsumexp(negative_logits.masked_fill(~negatives, -torch.inf), dim=1)
    ).mean()

    torch.testing.assert_close(result.loss, expected)
