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
