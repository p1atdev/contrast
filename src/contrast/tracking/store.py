from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import torch

import wandb
from contrast.config.loader import canonical_config_json
from contrast.config.schema import ExperimentConfig
from contrast.tracking.naming import wandb_run_name


class _WandbRun(Protocol):
    id: str | None
    entity: str | None
    project: str | None
    url: str | None

    def define_metric(
        self,
        name: str,
        *,
        step_metric: str | None = None,
        step_sync: bool | None = None,
        summary: str | None = None,
    ) -> object: ...

    def log(self, data: dict[str, Any]) -> None: ...

    def finish(self, exit_code: int | None = None) -> None: ...


@dataclass(frozen=True)
class _WandbReference:
    id: str
    entity: str | None
    project: str
    url: str | None

    @classmethod
    def read(cls, path: Path) -> _WandbReference:
        try:
            raw = json.loads(path.read_text())
        except FileNotFoundError as error:
            raise ValueError(f"W&B run metadata is missing: {path}") from error
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read W&B run metadata: {path}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"W&B run metadata must be a JSON object: {path}")

        run_id = _required_metadata_string(raw, "id", path)
        project = _required_metadata_string(raw, "project", path)
        entity = _optional_metadata_string(raw, "entity", path)
        url = _optional_metadata_string(raw, "url", path)
        return cls(id=run_id, entity=entity, project=project, url=url)

    def write(self, path: Path) -> None:
        contents = json.dumps(
            {
                "entity": self.entity,
                "id": self.id,
                "project": self.project,
                "url": self.url,
            },
            indent=2,
            sort_keys=True,
        )
        _atomic_write_text(path, contents + "\n")


def _required_metadata_string(raw: dict[str, Any], key: str, path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"W&B run metadata field '{key}' must be a non-empty string: {path}")
    return value


def _optional_metadata_string(raw: dict[str, Any], key: str, path: Path) -> str | None:
    value = raw.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"W&B run metadata field '{key}' must be a string or null: {path}")
    return value


def _atomic_write_text(destination: Path, contents: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(contents)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _git_value(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            ("git", *arguments),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_config(path: Path) -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate_json(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"run config is missing: {path}") from error
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read run config: {path}") from error


def _initialize_wandb_run(
    config: ExperimentConfig,
    directory: Path,
    *,
    display_name: str | None,
    run_id: str,
    project: str,
    entity: str | None,
    resume: Literal["allow", "never"],
) -> _WandbRun:
    initialization: dict[str, Any] = {
        "project": project,
        "entity": entity,
        "group": config.run.experiment,
        "config": config.model_dump(mode="json"),
        "dir": str(directory.resolve()),
        "mode": config.tracking.mode,
        "force": config.tracking.mode == "online",
        "id": run_id,
        "resume": resume,
    }
    if display_name is not None:
        initialization["name"] = display_name
    if resume == "never":
        initialization["tags"] = config.run.tags
    run = wandb.init(**initialization)
    if run is None:
        raise RuntimeError("wandb.init() did not return a run")
    return cast(_WandbRun, run)


def _reference_from_run(
    run: _WandbRun,
    *,
    requested_id: str,
    requested_project: str,
    requested_entity: str | None,
) -> _WandbReference:
    return _WandbReference(
        id=run.id or requested_id,
        entity=run.entity or requested_entity,
        project=run.project or requested_project,
        url=run.url,
    )


class RunStore:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        display_name: str | None = None,
    ) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"{timestamp}-{config.run.seed:04d}"
        self.directory = config.run.output_dir / config.run.experiment / run_name
        suffix = 1
        while self.directory.exists():
            self.directory = self.directory.with_name(f"{run_name}-{suffix}")
            suffix += 1
        self.checkpoint_directory = self.directory / "checkpoints"
        self.checkpoint_directory.mkdir(parents=True)
        (self.directory / "config.json").write_text(canonical_config_json(config))
        environment = {
            "created_at": datetime.now(UTC).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "command": sys.argv,
            "pid": os.getpid(),
        }
        (self.directory / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n"
        )

        run_id = uuid.uuid4().hex
        self._finished = False
        self._run = _initialize_wandb_run(
            config,
            self.directory,
            display_name=display_name or wandb_run_name(config),
            run_id=run_id,
            project=config.tracking.project,
            entity=config.tracking.entity,
            resume="never",
        )
        try:
            self._configure_metrics(config)
            _reference_from_run(
                self._run,
                requested_id=run_id,
                requested_project=config.tracking.project,
                requested_entity=config.tracking.entity,
            ).write(self.directory / "wandb.json")
        except BaseException:
            self.finish(exit_code=1)
            raise

    @classmethod
    def for_existing_run(cls, directory: str | Path) -> RunStore:
        resolved = Path(directory).resolve()
        config = _load_config(resolved / "config.json")
        reference = _WandbReference.read(resolved / "wandb.json")
        store = cls.__new__(cls)
        store.directory = resolved
        store.checkpoint_directory = resolved / "checkpoints"
        store._finished = False
        store._run = _initialize_wandb_run(
            config,
            resolved,
            display_name=None,
            run_id=reference.id,
            project=reference.project,
            entity=reference.entity,
            resume="allow",
        )
        try:
            store._configure_metrics(config)
            _reference_from_run(
                store._run,
                requested_id=reference.id,
                requested_project=reference.project,
                requested_entity=reference.entity,
            ).write(resolved / "wandb.json")
        except BaseException:
            store.finish(exit_code=1)
            raise
        return store

    def _configure_metrics(self, config: ExperimentConfig) -> None:
        self._run.define_metric("step")
        self._run.define_metric("*", step_metric="step", step_sync=False)
        selection_metric = config.evaluation.selection_metric
        if selection_metric is not None:
            self._run.define_metric(
                selection_metric,
                summary=config.evaluation.selection_mode,
            )

    def log(self, values: dict[str, Any]) -> None:
        if self._finished:
            raise RuntimeError("cannot log to a finished W&B run")
        event = {"time": datetime.now(UTC).isoformat(), **values}
        self._run.log(event)

    def finish(self, exit_code: int | None = 0) -> None:
        if self._finished:
            return
        self._finished = True
        self._run.finish(exit_code=exit_code)

    def save_checkpoint(self, state: dict[str, Any], name: str) -> Path:
        destination = self.checkpoint_directory / name
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.checkpoint_directory,
            prefix=f".{name}.",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(state, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        latest = self.checkpoint_directory / "latest.txt"
        latest.write_text(destination.name + "\n")
        return destination
