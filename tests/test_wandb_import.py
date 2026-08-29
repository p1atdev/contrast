import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from contrast.config.loader import load_experiment_config
from contrast.tracking.importer import ImportValidationError, import_runs


class FakeImportedRun:
    def __init__(self, *, url: str | None = "https://wandb.example/imported") -> None:
        self.url = url
        self.defined_metrics: list[tuple[str, dict[str, Any]]] = []
        self.logged: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}
        self.finished: list[int | None] = []

    def define_metric(self, name: str, **kwargs: Any) -> object:
        self.defined_metrics.append((name, kwargs))
        return object()

    def log(self, data: dict[str, Any]) -> None:
        self.logged.append(data.copy())

    def finish(self, exit_code: int | None = None) -> None:
        self.finished.append(exit_code)


def _write_local_run(
    root: Path,
    *,
    experiment: str = "historical-pilot",
    run_name: str = "20260827T000000Z-0003",
    metrics: list[dict[str, Any]] | None = None,
) -> Path:
    run_directory = root / experiment / run_name
    run_directory.mkdir(parents=True)
    config = load_experiment_config(
        "configs/base.toml",
        [
            f"run.experiment={experiment}",
            "run.seed=3",
            'run.tags=["vit-tiny","cifar100"]',
            "training.max_steps=20",
        ],
    )
    (run_directory / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
    (run_directory / "environment.json").write_text(
        json.dumps(
            {
                "created_at": "2026-08-27T00:00:00+00:00",
                "platform": "Linux-test",
                "python": "3.12-test",
                "torch": "2.8-test",
                "cuda": "13-test",
                "device": "Fake GPU",
                "git_commit": "deadbeef",
                "git_dirty": False,
                "hostname": "private-host",
                "pid": 12345,
                "command": ["python", "private-script.py"],
            }
        )
        + "\n"
    )
    source_metrics = metrics or [
        {
            "time": "2026-08-27T00:00:01+00:00",
            "type": "train",
            "epoch": 0,
            "step": 20,
            "loss": 2.5,
        },
        {
            "time": "2026-08-27T00:00:02+00:00",
            "type": "run_end",
            "epoch": 0,
            "step": 20,
            "completed": False,
        },
    ]
    (run_directory / "metrics.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in source_metrics)
    )
    return run_directory


def test_dry_run_validates_without_initializing_wandb_or_writing_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_directory = _write_local_run(tmp_path)

    def fail_init(**_: Any) -> None:
        raise AssertionError("dry-run must not initialize W&B")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fail_init))

    result = import_runs(tmp_path, project="target-project", dry_run=True)

    assert result.discovered == 1
    assert result.imported == 0
    assert result.skipped == 0
    assert result.events == 2
    assert not (run_directory / "wandb.json").exists()
    assert "pending=1" in capsys.readouterr().out


def test_import_maps_metadata_and_preserves_duplicate_steps_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = _write_local_run(tmp_path)
    remote_run = FakeImportedRun()
    init_calls: list[dict[str, Any]] = []

    def fake_init(**kwargs: Any) -> FakeImportedRun:
        init_calls.append(kwargs)
        return remote_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))

    first = import_runs(
        tmp_path,
        project="target-project",
        entity="research-team",
    )

    assert first.imported == 1
    assert first.events == 2
    assert len(init_calls) == 1
    initialization = init_calls[0]
    assert initialization["project"] == "target-project"
    assert initialization["entity"] == "research-team"
    assert initialization["name"] == "historical-pilot/supcon-seed-3"
    assert initialization["group"] == "historical-pilot"
    assert initialization["mode"] == "online"
    assert initialization["force"] is True
    assert initialization["resume"] == "never"
    assert initialization["tags"] == ["vit-tiny", "cifar100", "migrated", "pilot"]
    assert initialization["config"]["environment"] == {
        "created_at": "2026-08-27T00:00:00+00:00",
        "platform": "Linux-test",
        "python": "3.12-test",
        "torch": "2.8-test",
        "cuda": "13-test",
        "device": "Fake GPU",
        "git_commit": "deadbeef",
        "git_dirty": False,
    }
    assert "hostname" not in initialization["config"]["environment"]
    assert remote_run.defined_metrics == [
        ("step", {}),
        ("*", {"step_metric": "step", "step_sync": False}),
    ]
    assert [event["type"] for event in remote_run.logged] == ["train", "run_end"]
    assert [event["step"] for event in remote_run.logged] == [20, 20]
    assert remote_run.logged[0]["time"] == "2026-08-27T00:00:01+00:00"
    assert remote_run.summary["migration/status"] == "max_steps_reached"
    assert remote_run.summary["migration/event_count"] == 2
    assert remote_run.finished == [0]

    marker = json.loads((run_directory / "wandb.json").read_text())
    assert marker["id"] == initialization["id"]
    assert marker["project"] == "target-project"
    assert marker["entity"] == "research-team"
    assert marker["url"] == "https://wandb.example/imported"
    assert isinstance(marker["imported_at"], str)

    second = import_runs(tmp_path, project="target-project", entity="research-team")

    assert second.discovered == 1
    assert second.imported == 0
    assert second.skipped == 1
    assert second.events == 0
    assert len(init_calls) == 1


def test_existing_wandb_marker_skips_new_format_run_without_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = _write_local_run(tmp_path, experiment="new-format")
    (run_directory / "metrics.jsonl").unlink()
    (run_directory / "wandb.json").write_text(
        json.dumps(
            {
                "entity": None,
                "id": "already-tracked",
                "project": "target-project",
                "url": "https://wandb.example/already-tracked",
            }
        )
        + "\n"
    )

    def fail_init(**_: Any) -> None:
        raise AssertionError("an existing marker must prevent another W&B run")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fail_init))

    result = import_runs(tmp_path, project="target-project", dry_run=True)

    assert result.discovered == 1
    assert result.imported == 0
    assert result.skipped == 1
    assert result.events == 0


def test_import_failure_finishes_failed_run_without_writing_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = _write_local_run(tmp_path)

    class FailingRun(FakeImportedRun):
        def log(self, data: dict[str, Any]) -> None:
            super().log(data)
            raise RuntimeError("upload failed")

    remote_run = FailingRun()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **_: remote_run))

    with pytest.raises(RuntimeError, match="upload failed"):
        import_runs(tmp_path, project="target-project")

    assert remote_run.finished == [1]
    assert not (run_directory / "wandb.json").exists()


def test_import_preflight_rejects_bad_json_before_any_remote_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_local_run(tmp_path, experiment="a-valid")
    invalid = _write_local_run(tmp_path, experiment="z-invalid")
    (invalid / "metrics.jsonl").write_text(
        '{"time":"2026-08-27T00:00:00Z","type":"train","epoch":0,"step":1}\nnot-json\n'
    )
    init_calls: list[dict[str, Any]] = []
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **kwargs: init_calls.append(kwargs)),
    )

    with pytest.raises(ImportValidationError, match=r"metrics\.jsonl:2:"):
        import_runs(tmp_path, project="target-project")

    assert init_calls == []
