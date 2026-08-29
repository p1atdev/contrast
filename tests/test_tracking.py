import json
from pathlib import Path
from typing import Any

import pytest

import contrast.tracking.store as tracking_store
from contrast.config.loader import load_experiment_config
from contrast.tracking import RunStore
from contrast.tracking.naming import wandb_run_name


class FakeWandbRun:
    def __init__(
        self,
        *,
        run_id: str = "server-run-id",
        entity: str | None = "research-team",
        project: str = "tracked-project",
        url: str | None = "https://wandb.example/runs/server-run-id",
    ) -> None:
        self.id = run_id
        self.entity = entity
        self.project = project
        self.url = url
        self.defined_metrics: list[tuple[str, dict[str, Any]]] = []
        self.logged: list[dict[str, Any]] = []
        self.finished: list[int | None] = []

    def define_metric(self, name: str, **kwargs: Any) -> object:
        self.defined_metrics.append((name, kwargs))
        return object()

    def log(self, data: dict[str, Any]) -> None:
        self.logged.append(data.copy())

    def finish(self, exit_code: int | None = None) -> None:
        self.finished.append(exit_code)


def _config(tmp_path: Path, *overrides: str):
    return load_experiment_config(
        "configs/base.toml",
        [
            f"run.output_dir={json.dumps(str(tmp_path))}",
            "run.experiment=tracking-test",
            'run.tags=["unit-test","wandb"]',
            *overrides,
        ],
    )


def test_new_run_initializes_wandb_and_keeps_only_local_run_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run = FakeWandbRun()
    init_calls: list[dict[str, Any]] = []

    def fake_init(**kwargs: Any) -> FakeWandbRun:
        init_calls.append(kwargs)
        return fake_run

    monkeypatch.setattr(tracking_store.wandb, "init", fake_init)
    config = _config(
        tmp_path,
        "tracking.project=requested-project",
        "tracking.entity=research-team",
        "tracking.mode=offline",
    )

    store = RunStore(config)

    assert len(init_calls) == 1
    call = init_calls[0]
    assert call["project"] == "requested-project"
    assert call["entity"] == "research-team"
    assert call["group"] == "tracking-test"
    assert call["name"] == "tracking-test/supcon-seed-0"
    assert call["tags"] == ("unit-test", "wandb")
    assert call["config"]["tracking"]["mode"] == "offline"
    assert call["dir"] == str(store.directory.resolve())
    assert call["mode"] == "offline"
    assert call["force"] is False
    assert call["resume"] == "never"
    assert isinstance(call["id"], str) and call["id"]
    assert fake_run.defined_metrics == [
        ("step", {}),
        ("*", {"step_metric": "step", "step_sync": False}),
        ("eval/backbone_knn_top1", {"summary": "max"}),
    ]
    assert json.loads((store.directory / "wandb.json").read_text()) == {
        "entity": "research-team",
        "id": "server-run-id",
        "project": "tracked-project",
        "url": "https://wandb.example/runs/server-run-id",
    }
    assert (store.directory / "config.json").is_file()
    assert (store.directory / "environment.json").is_file()
    assert store.checkpoint_directory.is_dir()
    assert not (store.directory / "metrics.jsonl").exists()

    checkpoint = store.save_checkpoint({"epoch": 1}, "final.pt")

    assert checkpoint == store.checkpoint_directory / "final.pt"
    assert checkpoint.is_file()
    assert (store.checkpoint_directory / "latest.txt").read_text() == "final.pt\n"
    assert fake_run.logged == []


def test_run_name_uses_experiment_condition_and_seed(tmp_path: Path) -> None:
    config = _config(tmp_path, "run.seed=3")

    assert wandb_run_name(config) == "tracking-test/supcon-seed-3"

    sigmoid = _config(tmp_path, "run.seed=3", "objective.kind=sigmoid_supcon")
    assert wandb_run_name(sigmoid) == "tracking-test/sigmoid-supcon-seed-3"


def test_log_preserves_duplicate_source_steps_as_separate_history_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run = FakeWandbRun()
    monkeypatch.setattr(tracking_store.wandb, "init", lambda **_: fake_run)
    store = RunStore(_config(tmp_path, "tracking.mode=disabled"))

    store.log({"type": "train", "step": 20, "epoch": 0, "loss": 2.5})
    store.log(
        {
            "type": "evaluation",
            "step": 20,
            "epoch": 0,
            "eval/backbone_knn_top1": 0.4,
        }
    )

    assert len(fake_run.logged) == 2
    assert [event["type"] for event in fake_run.logged] == ["train", "evaluation"]
    assert [event["step"] for event in fake_run.logged] == [20, 20]
    assert all("time" in event for event in fake_run.logged)


def test_finish_is_idempotent_and_prevents_late_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run = FakeWandbRun()
    monkeypatch.setattr(tracking_store.wandb, "init", lambda **_: fake_run)
    store = RunStore(_config(tmp_path, "tracking.mode=disabled"))

    store.finish(exit_code=7)
    store.finish(exit_code=0)

    assert fake_run.finished == [7]
    with pytest.raises(RuntimeError, match="finished W&B run"):
        store.log({"type": "train", "step": 1, "epoch": 0})


def test_existing_run_resumes_using_wandb_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = tmp_path / "existing"
    run_directory.mkdir()
    config = _config(tmp_path, "tracking.mode=disabled")
    (run_directory / "config.json").write_text(config.model_dump_json())
    (run_directory / "wandb.json").write_text(
        json.dumps(
            {
                "entity": "original-team",
                "id": "original-id",
                "project": "original-project",
                "url": None,
            }
        )
    )
    fake_run = FakeWandbRun(
        run_id="original-id",
        entity="original-team",
        project="original-project",
    )
    init_calls: list[dict[str, Any]] = []

    def fake_init(**kwargs: Any) -> FakeWandbRun:
        init_calls.append(kwargs)
        return fake_run

    monkeypatch.setattr(tracking_store.wandb, "init", fake_init)

    store = RunStore.for_existing_run(run_directory)

    assert store.directory == run_directory.resolve()
    assert init_calls[0]["id"] == "original-id"
    assert init_calls[0]["project"] == "original-project"
    assert init_calls[0]["entity"] == "original-team"
    assert init_calls[0]["resume"] == "allow"
    assert "tags" not in init_calls[0]
    assert fake_run.defined_metrics == [
        ("step", {}),
        ("*", {"step_metric": "step", "step_sync": False}),
        ("eval/backbone_knn_top1", {"summary": "max"}),
    ]


def test_existing_run_requires_wandb_reference_before_initializing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, "tracking.mode=disabled")
    (tmp_path / "config.json").write_text(config.model_dump_json())
    init_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(tracking_store.wandb, "init", lambda **kwargs: init_calls.append(kwargs))

    with pytest.raises(ValueError, match="metadata is missing"):
        RunStore.for_existing_run(tmp_path)

    assert init_calls == []
