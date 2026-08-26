import json
from pathlib import Path

from contrast.cli import _config_from_checkpoint, build_parser
from contrast.config.loader import load_experiment_config
from contrast.tracking import RunStore


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
        ["evaluate", "--checkpoint", "runs/example/checkpoints/final.pt"]
    )

    assert arguments.command == "evaluate"
    assert arguments.checkpoint.endswith("final.pt")


def test_existing_run_store_appends_without_reinitializing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n")

    store = RunStore.for_existing_run(tmp_path)
    store.log({"type": "offline_evaluation", "step": 10, "epoch": 2})

    assert config_path.read_text() == "{}\n"
    event = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert event["type"] == "offline_evaluation"
    assert event["step"] == 10
