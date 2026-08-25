from contrast.config.schema import (
    CrossEntropyObjectiveConfig,
    JointObjectiveConfig,
    NTXentObjectiveConfig,
    ObjectiveConfig,
    SigmoidSupConObjectiveConfig,
    SincereObjectiveConfig,
    SupConObjectiveConfig,
)
from contrast.objectives.base import Objective
from contrast.objectives.losses import (
    CrossEntropyObjective,
    JointObjective,
    SigmoidSupConObjective,
    SincereObjective,
    SoftmaxContrastiveObjective,
)


def build_objective(config: ObjectiveConfig) -> Objective:
    if isinstance(config, CrossEntropyObjectiveConfig):
        return CrossEntropyObjective()
    if isinstance(config, NTXentObjectiveConfig):
        return SoftmaxContrastiveObjective(config.temperature, "instance")
    if isinstance(config, SupConObjectiveConfig):
        return SoftmaxContrastiveObjective(config.temperature, "class")
    if isinstance(config, SincereObjectiveConfig):
        return SincereObjective(config.temperature)
    if isinstance(config, SigmoidSupConObjectiveConfig):
        return SigmoidSupConObjective(config.scale_init, config.bias_init)
    if isinstance(config, JointObjectiveConfig):
        return JointObjective(
            config.temperature,
            config.cross_entropy_weight,
            config.contrastive_weight,
        )
    raise TypeError(f"unsupported objective config: {type(config).__name__}")
