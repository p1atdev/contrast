import pytest
import torch

from contrast.models.model import ModelOutput
from contrast.objectives.base import ObjectiveMetadata
from contrast.objectives.losses import (
    CrossEntropyObjective,
    SigmoidSupConObjective,
    SincereObjective,
    SoftmaxContrastiveObjective,
)


@pytest.fixture
def problem() -> tuple[ModelOutput, ObjectiveMetadata]:
    torch.manual_seed(2)
    features = torch.randn(6, 8, requires_grad=True)
    embeddings = torch.nn.functional.normalize(torch.randn(6, 5), dim=-1).requires_grad_()
    logits = torch.randn(6, 3, requires_grad=True)
    metadata = ObjectiveMetadata(
        labels=torch.tensor([0, 0, 1, 1, 2, 2]),
        source_ids=torch.tensor([0, 0, 1, 1, 2, 2]),
        view_ids=torch.tensor([0, 1, 0, 1, 0, 1]),
    )
    return ModelOutput(features, embeddings, logits), metadata


@pytest.mark.parametrize(
    "objective",
    [
        CrossEntropyObjective(),
        SoftmaxContrastiveObjective(0.1, "instance"),
        SoftmaxContrastiveObjective(0.1, "class"),
        SincereObjective(0.1),
        SigmoidSupConObjective(10.0, "auto"),
    ],
)
def test_objective_is_finite_and_differentiable(
    objective: torch.nn.Module,
    problem: tuple[ModelOutput, ObjectiveMetadata],
) -> None:
    output, metadata = problem
    result = objective(output, metadata)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert any(tensor.grad is not None for tensor in output.as_dict().values())


def _uneven_positive_problem() -> tuple[ModelOutput, ObjectiveMetadata]:
    torch.manual_seed(11)
    embeddings = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    output = ModelOutput(
        features=torch.randn(5, 3),
        embeddings=embeddings,
        logits=torch.randn(5, 2),
    )
    metadata = ObjectiveMetadata(
        labels=torch.tensor([0, 0, 0, 1, 1]),
        source_ids=torch.arange(5),
        view_ids=torch.zeros(5, dtype=torch.long),
    )
    return output, metadata


def test_sincere_weights_each_anchor_equally() -> None:
    output, metadata = _uneven_positive_problem()
    objective = SincereObjective(temperature=0.2)
    result = objective(output, metadata)

    similarities = output.embeddings @ output.embeddings.T / 0.2
    valid = ~torch.eye(5, dtype=torch.bool)
    positives = metadata.labels[:, None].eq(metadata.labels[None, :]) & valid
    negatives = valid & ~positives
    negative_lse = torch.logsumexp(
        similarities.masked_fill(~negatives, -torch.inf),
        dim=1,
    )
    pair_losses = torch.logaddexp(similarities, negative_lse[:, None]) - similarities
    counts = positives.sum(dim=1)
    expected = (torch.where(positives, pair_losses, 0.0).sum(dim=1) / counts).mean()
    pair_weighted = pair_losses.masked_select(positives).mean()

    torch.testing.assert_close(result.loss, expected)
    assert not torch.isclose(result.loss, pair_weighted)


def test_sigmoid_supcon_normalizes_each_anchor_by_positive_count() -> None:
    output, metadata = _uneven_positive_problem()
    objective = SigmoidSupConObjective(scale_init=10.0, bias_init=-10.0)
    result = objective(output, metadata)

    valid = ~torch.eye(5, dtype=torch.bool)
    positives = metadata.labels[:, None].eq(metadata.labels[None, :]) & valid
    logits = objective.log_scale.exp() * (output.embeddings @ output.embeddings.T)
    logits = logits + objective.bias
    signs = torch.where(positives, 1.0, -1.0)
    pair_losses = torch.nn.functional.softplus(-signs * logits)
    counts = positives.sum(dim=1)
    expected = (torch.where(valid, pair_losses, 0.0).sum(dim=1) / counts).mean()

    torch.testing.assert_close(result.loss, expected)
