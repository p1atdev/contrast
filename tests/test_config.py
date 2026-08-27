from pathlib import Path

import pytest
from pydantic import ValidationError

from contrast.config.loader import load_experiment_config


def test_objective_override_replaces_discriminated_table() -> None:
    config = load_experiment_config(Path("configs/objectives/sigmoid_supcon.toml"))
    assert config.objective.kind == "sigmoid_supcon"
    assert config.data.huggingface.repo_id == "uoft-cs/cifar100"
    assert len(config.data.huggingface.revision) == 40
    assert config.objective.scale_init == 10.0
    assert config.objective.bias_init == -10.0
    assert config.optimizer.weight_decay_policy == "standard"
    assert config.training.epochs == 120
    assert config.training.gradient_clip_norm == 10.0
    assert config.evaluation.selection_metric == "eval/backbone_knn_top1"
    assert not config.evaluation.test_at_end


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text("schema_version = 1\nunknown = true\n")
    with pytest.raises(ValidationError):
        load_experiment_config(path)


def test_dotted_override_is_validated() -> None:
    config = load_experiment_config(
        "configs/base.toml",
        [
            "run.seed=7",
            "batch.global_source_batch_size=64",
            "batch.grad_cache_chunk_size_per_rank=32",
        ],
    )
    assert config.run.seed == 7
    assert config.data.split_seed == 0
    assert config.batch.global_source_batch_size == 64
    assert config.evaluation.knn_spaces == ("backbone", "projector")


def test_dotted_override_can_replace_ema_decay_schedule() -> None:
    config = load_experiment_config(
        "configs/base.toml",
        ["ema.decay.kind=constant", "ema.decay.decay=0.95"],
    )

    assert config.ema.enabled
    assert config.ema.evaluation_weights == "both"
    assert config.ema.decay.kind == "constant"
    assert config.ema.decay.decay == 0.95
