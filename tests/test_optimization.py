import torch
from torch import nn

from contrast.config.schema import AdamWScheduleFreeOptimizerConfig
from contrast.training.optimization import OptimizationController


def _parameters() -> list[tuple[str, nn.Parameter]]:
    matrix = nn.Parameter(torch.ones(2, 2))
    bias = nn.Parameter(torch.ones(2))
    special = nn.Parameter(torch.ones(1, 2, 2))
    special._no_weight_decay = True
    scalar = nn.Parameter(torch.tensor(1.0))
    return [
        ("model.matrix", matrix),
        ("model.bias", bias),
        ("model.special", special),
        ("objective.scalar", scalar),
    ]


def test_standard_weight_decay_excludes_bias_norm_special_and_scalar() -> None:
    named_parameters = _parameters()
    controller = OptimizationController(
        AdamWScheduleFreeOptimizerConfig(
            weight_decay=0.05,
            weight_decay_policy="standard",
        ),
        named_parameters,
        steps=10,
    )

    decay_group, no_decay_group = controller.optimizer.param_groups
    assert decay_group["weight_decay"] == 0.05
    assert no_decay_group["weight_decay"] == 0.0
    assert {id(parameter) for parameter in decay_group["params"]} == {id(named_parameters[0][1])}
    assert {id(parameter) for parameter in no_decay_group["params"]} == {
        id(parameter) for _, parameter in named_parameters[1:]
    }


def test_schedule_free_lr_reports_warmup_adjusted_value() -> None:
    named_parameters = _parameters()
    controller = OptimizationController(
        AdamWScheduleFreeOptimizerConfig(lr=0.001, warmup_steps=4),
        named_parameters,
        steps=10,
    )
    controller.train()
    for _, parameter in named_parameters:
        parameter.grad = torch.ones_like(parameter)

    controller.step()

    group = controller.optimizer.param_groups[0]
    assert group["lr"] == 0.001
    assert group["scheduled_lr"] == 0.00025
    assert controller.lr == group["scheduled_lr"]


def test_legacy_single_group_optimizer_state_migrates() -> None:
    legacy_parameters = _parameters()
    legacy = OptimizationController(
        AdamWScheduleFreeOptimizerConfig(
            weight_decay=0.05,
            weight_decay_policy="all",
        ),
        legacy_parameters,
        steps=10,
    )
    legacy.train()
    for _, parameter in legacy_parameters:
        parameter.grad = torch.ones_like(parameter)
    legacy.step()
    state = legacy.state_dict()
    assert len(state["optimizer"]["param_groups"]) == 1

    current = OptimizationController(
        AdamWScheduleFreeOptimizerConfig(
            weight_decay=0.05,
            weight_decay_policy="standard",
        ),
        _parameters(),
        steps=10,
    )
    current.load_state_dict(state)

    assert len(current.optimizer.param_groups) == 2
    assert [group["weight_decay"] for group in current.optimizer.param_groups] == [0.05, 0.0]
    assert len(current.optimizer.state) == 4
