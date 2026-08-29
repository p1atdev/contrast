import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import contrast.cli as cli
from contrast.config.loader import load_experiment_config


class FakeStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.finished: list[int | None] = []
        self.logged: list[dict[str, Any]] = []

    def finish(self, exit_code: int | None = 0) -> None:
        self.finished.append(exit_code)

    def log(self, values: dict[str, Any]) -> None:
        self.logged.append(values.copy())


class FakeRuntime:
    def __init__(self) -> None:
        self.distributed = False
        self.is_primary = False
        self.device = torch.device("cpu")
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _config(tmp_path: Path):
    return load_experiment_config(
        "configs/base.toml",
        [
            f"run.output_dir={json.dumps(str(tmp_path))}",
            "tracking.mode=disabled",
            "ema.enabled=false",
        ],
    )


@pytest.mark.parametrize("fit_fails", [False, True])
def test_train_always_finishes_wandb_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fit_fails: bool,
) -> None:
    config = _config(tmp_path)
    runtime = FakeRuntime()
    store = FakeStore(tmp_path / "run")

    class FakePrecision:
        def configure_backends(self, _reproducibility: object) -> None:
            pass

    class FakeTrainer:
        def __init__(self, *_: object) -> None:
            pass

        def fit(self) -> Path:
            if fit_fails:
                raise RuntimeError("training failed")
            return tmp_path / "run" / "checkpoints" / "final.pt"

    monkeypatch.setattr(cli, "load_experiment_config", lambda *_: config)
    monkeypatch.setattr(cli.RuntimeContext, "initialize", lambda: runtime)
    monkeypatch.setattr(cli, "seed_everything", lambda *_: None)
    monkeypatch.setattr(cli, "PrecisionManager", lambda *_: FakePrecision())
    monkeypatch.setattr(cli, "build_cifar100_loaders", lambda *_: object())
    monkeypatch.setattr(cli, "build_model", lambda *_: object())
    monkeypatch.setattr(cli, "build_objective", lambda *_: object())
    monkeypatch.setattr(cli, "RunStore", lambda *_, **__: store)
    monkeypatch.setattr(cli, "Trainer", FakeTrainer)
    arguments = argparse.Namespace(config="config.toml", overrides=[], resume=None)

    if fit_fails:
        with pytest.raises(RuntimeError, match="training failed"):
            cli._train(arguments)
        expected_exit_code = 1
    else:
        assert cli._train(arguments) == 0
        expected_exit_code = 0

    assert store.finished == [expected_exit_code]
    assert runtime.closed == 1


def test_offline_evaluation_finishes_resumed_run_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    state = {
        "config": config.model_dump(mode="json"),
        "model": {},
        "epoch": 4,
        "global_step": 80,
    }
    runtime = FakeRuntime()
    store = FakeStore(tmp_path / "run")

    class FakePrecision:
        def configure_backends(self, _reproducibility: object) -> None:
            pass

    class FakeModel:
        def to(self, _device: torch.device) -> "FakeModel":
            return self

        def load_state_dict(self, _state: dict[str, Any]) -> None:
            pass

    monkeypatch.setattr(cli.torch, "load", lambda *_args, **_kwargs: state)
    monkeypatch.setattr(
        cli,
        "RunStore",
        SimpleNamespace(for_existing_run=lambda _directory: store),
    )
    monkeypatch.setattr(cli.RuntimeContext, "initialize", lambda: runtime)
    monkeypatch.setattr(cli, "seed_everything", lambda *_: None)
    monkeypatch.setattr(cli, "PrecisionManager", lambda *_: FakePrecision())
    monkeypatch.setattr(
        cli,
        "build_cifar100_loaders",
        lambda *_: SimpleNamespace(memory="memory", validation="validation", test="test"),
    )
    monkeypatch.setattr(cli, "build_model", lambda *_: FakeModel())
    monkeypatch.setattr(
        cli,
        "evaluate",
        lambda *_args, **_kwargs: {"eval/backbone_knn_top1": 0.5},
    )
    arguments = argparse.Namespace(
        checkpoint=str(tmp_path / "run" / "checkpoints" / "best.pt"),
        run_dir=str(tmp_path / "run"),
        queries="eval",
    )

    assert cli._evaluate_checkpoint(arguments) == 0

    assert store.finished == [0]
    assert runtime.closed == 1
    assert store.logged == [
        {
            "type": "offline_evaluation",
            "epoch": 4,
            "step": 80,
            "checkpoint": str(Path(arguments.checkpoint).resolve()),
            "eval/backbone_knn_top1": 0.5,
        }
    ]


def test_offline_evaluation_finishes_resumed_run_when_runtime_initialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    state = {"config": config.model_dump(mode="json")}
    store = FakeStore(tmp_path / "run")
    monkeypatch.setattr(cli.torch, "load", lambda *_args, **_kwargs: state)
    monkeypatch.setattr(
        cli,
        "RunStore",
        SimpleNamespace(for_existing_run=lambda _directory: store),
    )
    monkeypatch.setattr(
        cli.RuntimeContext,
        "initialize",
        lambda: (_ for _ in ()).throw(RuntimeError("runtime failed")),
    )
    arguments = argparse.Namespace(
        checkpoint=str(tmp_path / "run" / "checkpoints" / "best.pt"),
        run_dir=str(tmp_path / "run"),
        queries="eval",
    )

    with pytest.raises(RuntimeError, match="runtime failed"):
        cli._evaluate_checkpoint(arguments)

    assert store.finished == [1]
