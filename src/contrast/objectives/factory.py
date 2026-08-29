from contrast.config.schema import (
    ArcFaceObjectiveConfig,
    BarlowTwinsObjectiveConfig,
    BatchHardTripletObjectiveConfig,
    BYOLObjectiveConfig,
    CircleObjectiveConfig,
    CosFaceObjectiveConfig,
    CrossEntropyObjectiveConfig,
    JointObjectiveConfig,
    MoCoObjectiveConfig,
    ModelConfig,
    MultiSimilarityObjectiveConfig,
    NormalizedSoftmaxObjectiveConfig,
    NTXentObjectiveConfig,
    ObjectiveConfig,
    ProxyAnchorObjectiveConfig,
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
from contrast.objectives.metric_losses import (
    ArcFaceObjective,
    BatchHardTripletObjective,
    CircleLossObjective,
    CosFaceObjective,
    MultiSimilarityObjective,
    NormalizedSoftmaxObjective,
    ProxyAnchorObjective,
)
from contrast.objectives.self_supervised import (
    BarlowTwinsObjective,
    BYOLObjective,
    MoCoObjective,
)


def _space(configured: str, model: ModelConfig) -> tuple[str, int]:
    if configured == "backbone":
        return "features", model.dim
    if configured == "projector":
        return "embeddings", model.projection.output_dim
    raise ValueError(f"unsupported objective space: {configured}")


def build_objective(config: ObjectiveConfig, model: ModelConfig) -> Objective:
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
    if isinstance(config, NormalizedSoftmaxObjectiveConfig):
        source, dimension = _space(config.space, model)
        return NormalizedSoftmaxObjective(
            model.num_classes,
            dimension,
            config.scale,
            feature_source=source,
        )
    if isinstance(config, CosFaceObjectiveConfig):
        source, dimension = _space(config.space, model)
        return CosFaceObjective(
            model.num_classes,
            dimension,
            config.scale,
            config.margin,
            feature_source=source,
        )
    if isinstance(config, ArcFaceObjectiveConfig):
        source, dimension = _space(config.space, model)
        return ArcFaceObjective(
            model.num_classes,
            dimension,
            config.scale,
            config.margin,
            feature_source=source,
        )
    if isinstance(config, CircleObjectiveConfig):
        return CircleLossObjective(config.scale, config.margin)
    if isinstance(config, ProxyAnchorObjectiveConfig):
        source, dimension = _space(config.space, model)
        return ProxyAnchorObjective(
            model.num_classes,
            dimension,
            config.scale,
            config.margin,
            feature_source=source,
        )
    if isinstance(config, BatchHardTripletObjectiveConfig):
        return BatchHardTripletObjective(config.margin)
    if isinstance(config, MultiSimilarityObjectiveConfig):
        return MultiSimilarityObjective(
            config.alpha,
            config.beta,
            config.base,
            config.mining_epsilon,
        )
    if isinstance(config, BarlowTwinsObjectiveConfig):
        return BarlowTwinsObjective(config.redundancy_weight, config.eps)
    if isinstance(config, BYOLObjectiveConfig):
        return BYOLObjective(
            model.projection.output_dim,
            config.predictor_hidden_dim,
            config.base_target_decay,
            config.final_target_decay,
        )
    if isinstance(config, MoCoObjectiveConfig):
        return MoCoObjective(
            model.projection.output_dim,
            config.queue_size,
            config.temperature,
            config.target_decay,
            config.symmetric,
        )
    raise TypeError(f"unsupported objective config: {type(config).__name__}")
