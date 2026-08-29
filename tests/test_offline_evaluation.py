from pathlib import Path

import pytest
from pydantic import ValidationError

from contrast.cli import _config_from_checkpoint, _expand_sweep, build_parser
from contrast.config.loader import load_experiment_config


def test_legacy_checkpoint_uses_original_run_seed_for_split() -> None:
    raw = load_experiment_config(
        "configs/base.toml",
        ["run.seed=7"],
    ).model_dump(mode="json")
    del raw["ema"]
    del raw["data"]["split_seed"]

    config = _config_from_checkpoint({"config": raw})

    assert not config.ema.enabled
    assert config.run.seed == 7
    assert config.data.split_seed == 7


def test_current_checkpoint_preserves_explicit_split_seed() -> None:
    raw = load_experiment_config(
        "configs/base.toml",
        ["run.seed=7", "data.split_seed=3"],
    ).model_dump(mode="json")

    config = _config_from_checkpoint({"config": raw})

    assert config.data.split_seed == 3

    assert config.ema.enabled


def test_evaluate_subcommand_accepts_checkpoint() -> None:
    arguments = build_parser().parse_args(
        [
            "evaluate",
            "--checkpoint",
            "runs/example/checkpoints/final.pt",
            "--queries",
            "test",
        ]
    )

    assert arguments.command == "evaluate"
    assert arguments.checkpoint.endswith("final.pt")
    assert arguments.queries == "test"


def test_wandb_import_subcommand_accepts_destination_and_dry_run() -> None:
    arguments = build_parser().parse_args(
        [
            "wandb-import",
            "--runs-dir",
            "historical-runs",
            "--project",
            "contrast-history",
            "--entity",
            "research-team",
            "--dry-run",
        ]
    )

    assert arguments.command == "wandb-import"
    assert arguments.runs_dir == "historical-runs"
    assert arguments.project == "contrast-history"
    assert arguments.entity == "research-team"
    assert arguments.dry_run is True


def test_core_sweep_expands_seed_major() -> None:
    runs = _expand_sweep(Path("configs/sweeps/core_losses.toml").resolve())

    assert len(runs) == 15
    expected_names = [
        "ce.toml",
        "ntxent.toml",
        "supcon.toml",
        "sincere.toml",
        "sigmoid_supcon.toml",
    ]
    assert [path.name for path, _ in runs[:5]] == expected_names
    assert all("run.seed=0" in overrides for _, overrides in runs[:5])
    assert all("run.seed=1" in overrides for _, overrides in runs[5:10])
    assert all('run.experiment="cifar100-core-v3"' in overrides for _, overrides in runs)


def test_sweep_preflight_rejects_invalid_later_combination(tmp_path: Path) -> None:
    base = Path("configs/base.toml").resolve()
    sweep = tmp_path / "invalid.toml"
    sweep.write_text(f'base = "{base}"\n\n[grid]\n"run.seed" = [0, 1]\n"model.unknown" = [true]\n')

    with pytest.raises(ValidationError):
        _expand_sweep(sweep)
