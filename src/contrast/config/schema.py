from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(FrozenModel):
    experiment: str = "cifar100-core"
    output_dir: Path = Path("runs")
    seed: int = 0
    tags: tuple[str, ...] = ()


class AugmentationConfig(FrozenModel):
    crop_padding: int = Field(default=4, ge=0)
    horizontal_flip_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    color_jitter_strength: float = Field(default=0.5, ge=0.0)
    grayscale_probability: float = Field(default=0.2, ge=0.0, le=1.0)


class HuggingFaceDataConfig(FrozenModel):
    repo_id: str = "uoft-cs/cifar100"
    revision: str = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"


class DataConfig(FrozenModel):
    name: Literal["cifar100"] = "cifar100"
    root: Path = Path("data")
    download: bool = True
    huggingface: HuggingFaceDataConfig = HuggingFaceDataConfig()
    validation_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    augmentation: AugmentationConfig = AugmentationConfig()


class BatchConfig(FrozenModel):
    global_source_batch_size: PositiveInt = 256
    views: PositiveInt = 2
    grad_cache_chunk_size_per_rank: PositiveInt = 32

    @model_validator(mode="after")
    def validate_sizes(self) -> BatchConfig:
        if self.views < 2:
            raise ValueError("contrastive objectives require at least two views")
        if self.grad_cache_chunk_size_per_rank > self.global_source_batch_size:
            raise ValueError("GradCache chunk cannot exceed the global source batch")
        return self


class LayerConfig(FrozenModel):
    normalization: Literal["layer_norm", "rms_norm"] = "layer_norm"
    activation: Literal["gelu", "silu", "relu"] = "gelu"
    attention: Literal["eager", "sdpa"] = "eager"
    norm_position: Literal["pre"] = "pre"
    norm_eps: float = Field(default=1e-6, gt=0.0)


class ProjectionConfig(FrozenModel):
    hidden_dim: PositiveInt = 768
    output_dim: PositiveInt = 128


class ModelConfig(FrozenModel):
    name: Literal["vit"] = "vit"
    image_size: PositiveInt = 32
    patch_size: PositiveInt = 4
    dim: PositiveInt = 192
    depth: PositiveInt = 12
    num_heads: PositiveInt = 3
    mlp_ratio: float = Field(default=4.0, gt=0.0)
    num_classes: PositiveInt = 100
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    attention_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    layers: LayerConfig = LayerConfig()
    projection: ProjectionConfig = ProjectionConfig()

    @model_validator(mode="after")
    def validate_geometry(self) -> ModelConfig:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.dim % self.num_heads:
            raise ValueError("model dim must be divisible by num_heads")
        return self


class CrossEntropyObjectiveConfig(FrozenModel):
    kind: Literal["ce"] = "ce"


class NTXentObjectiveConfig(FrozenModel):
    kind: Literal["ntxent"] = "ntxent"
    temperature: float = Field(default=0.1, gt=0.0)


class SupConObjectiveConfig(FrozenModel):
    kind: Literal["supcon"] = "supcon"
    temperature: float = Field(default=0.1, gt=0.0)


class SincereObjectiveConfig(FrozenModel):
    kind: Literal["sincere"] = "sincere"
    temperature: float = Field(default=0.1, gt=0.0)


class SigmoidSupConObjectiveConfig(FrozenModel):
    kind: Literal["sigmoid_supcon"] = "sigmoid_supcon"
    scale_init: float = Field(default=10.0, gt=0.0)
    bias_init: float | Literal["auto"] = "auto"


class JointObjectiveConfig(FrozenModel):
    kind: Literal["ce_supcon"] = "ce_supcon"
    temperature: float = Field(default=0.1, gt=0.0)
    contrastive_weight: float = Field(default=1.0, ge=0.0)
    cross_entropy_weight: float = Field(default=1.0, ge=0.0)


ObjectiveConfig = Annotated[
    CrossEntropyObjectiveConfig
    | NTXentObjectiveConfig
    | SupConObjectiveConfig
    | SincereObjectiveConfig
    | SigmoidSupConObjectiveConfig
    | JointObjectiveConfig,
    Field(discriminator="kind"),
]


class CosineSchedulerConfig(FrozenModel):
    kind: Literal["cosine"] = "cosine"
    warmup_steps: int = Field(default=0, ge=0)
    minimum_lr_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class AdamWOptimizerConfig(FrozenModel):
    kind: Literal["adamw"] = "adamw"
    lr: float = Field(default=3e-4, gt=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(default=1e-8, gt=0.0)
    weight_decay: float = Field(default=0.05, ge=0.0)
    scheduler: CosineSchedulerConfig = CosineSchedulerConfig()


class AdamWScheduleFreeOptimizerConfig(FrozenModel):
    kind: Literal["adamw_schedule_free"] = "adamw_schedule_free"
    lr: float = Field(default=1e-3, gt=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(default=1e-8, gt=0.0)
    weight_decay: float = Field(default=0.05, ge=0.0)
    warmup_steps: int = Field(default=500, ge=0)


OptimizerConfig = Annotated[
    AdamWOptimizerConfig | AdamWScheduleFreeOptimizerConfig,
    Field(discriminator="kind"),
]


class PrecisionConfig(FrozenModel):
    parameter_dtype: Literal["float32"] = "float32"
    autocast_dtype: Literal["none", "bfloat16", "float16"] = "bfloat16"
    loss_dtype: Literal["float32"] = "float32"
    allow_tf32: bool = True
    matmul_precision: Literal["highest", "high", "medium"] = "high"


class ReproducibilityConfig(FrozenModel):
    mode: Literal["seeded", "strict", "fast"] = "seeded"
    cudnn_benchmark: bool = False


class TrainingConfig(FrozenModel):
    epochs: PositiveInt = 400
    step_strategy: Literal["direct", "grad_cache"] = "grad_cache"
    log_every_steps: PositiveInt = 20
    evaluate_every_epochs: PositiveInt = 1
    checkpoint_every_epochs: PositiveInt = 1
    gradient_clip_norm: float | None = Field(default=1.0, gt=0.0)
    max_steps: PositiveInt | None = None


class EvaluationConfig(FrozenModel):
    enabled: bool = True
    knn_k: PositiveInt = 20


class ExperimentConfig(FrozenModel):
    schema_version: Literal[1] = 1
    run: RunConfig = RunConfig()
    data: DataConfig = DataConfig()
    batch: BatchConfig = BatchConfig()
    model: ModelConfig = ModelConfig()
    objective: ObjectiveConfig = SupConObjectiveConfig()
    optimizer: OptimizerConfig = AdamWScheduleFreeOptimizerConfig()
    precision: PrecisionConfig = PrecisionConfig()
    reproducibility: ReproducibilityConfig = ReproducibilityConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
