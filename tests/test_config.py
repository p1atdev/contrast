from pathlib import Path

import pytest
from pydantic import ValidationError

from contrast.config.loader import load_experiment_config
from contrast.objectives import build_objective


def test_objective_override_replaces_discriminated_table() -> None:
    config = load_experiment_config(Path("configs/objectives/sigmoid_supcon.toml"))
    assert config.objective.kind == "sigmoid_supcon"
    assert config.data.huggingface.repo_id == "uoft-cs/cifar100"
    assert len(config.data.huggingface.revision) == 40
    assert config.objective.scale_init == 10.0
    assert config.objective.bias_init == -10.0
    assert config.optimizer.weight_decay_policy == "standard"
    assert config.training.epochs == 120
    assert config.training.gradient_clip_norm == 100.0
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


def test_tracking_defaults_to_online_contrast_lab_project() -> None:
    config = load_experiment_config("configs/base.toml")

    assert config.tracking.project == "contrast-lab"
    assert config.tracking.entity is None
    assert config.tracking.mode == "online"


def test_tracking_can_be_disabled_or_sent_to_an_entity() -> None:
    disabled = load_experiment_config(
        "configs/base.toml",
        ["tracking.mode=disabled"],
    )
    team = load_experiment_config(
        "configs/base.toml",
        ["tracking.entity=research-team", "tracking.project=contrast-comparisons"],
    )

    assert disabled.tracking.mode == "disabled"
    assert team.tracking.entity == "research-team"
    assert team.tracking.project == "contrast-comparisons"


@pytest.mark.parametrize(
    "name",
    [
        "normalized_softmax",
        "cosface",
        "arcface",
        "circle",
        "proxy_anchor",
        "batch_hard_triplet",
        "multi_similarity",
        "barlow_twins",
        "byol",
        "moco",
    ],
)
def test_extended_objective_configs_build(name: str) -> None:
    config = load_experiment_config(f"configs/objectives/{name}.toml")

    assert config.objective.kind == name
    assert build_objective(config.objective, config.model) is not None


@pytest.mark.parametrize("name", ["barlow_twins", "byol", "moco"])
def test_two_view_self_supervised_objectives_reject_extra_views(name: str) -> None:
    with pytest.raises(ValidationError, match=r"requires batch\.views=2"):
        load_experiment_config(
            f"configs/objectives/{name}.toml",
            ["batch.views=3"],
        )
