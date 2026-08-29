from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from contrast.config.schema import ExperimentConfig
from contrast.tracking.naming import wandb_run_name

_ENVIRONMENT_KEYS = frozenset(
    {
        "created_at",
        "platform",
        "python",
        "torch",
        "cuda",
        "device",
        "git_commit",
        "git_dirty",
    }
)
_RUN_ID_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


class ImportValidationError(ValueError):
    """A local run cannot be safely imported."""


@dataclass(frozen=True)
class LocalRun:
    directory: Path
    config: dict[str, Any]
    environment: dict[str, Any]
    metrics: tuple[dict[str, Any], ...]
    experiment: str
    display_name: str
    tags: tuple[str, ...]
    max_steps: int | None
    run_id: str
    already_imported: bool

    @property
    def marker_path(self) -> Path:
        return self.directory / "wandb.json"


@dataclass(frozen=True)
class ImportResult:
    discovered: int
    imported: int
    skipped: int
    events: int


def _validation_error(path: Path, line: int, message: str) -> ImportValidationError:
    return ImportValidationError(f"{path}:{line}: {message}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _ensure_finite(value: Any, path: Path, line: int) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _validation_error(path, line, "non-finite number")
    if isinstance(value, dict):
        for child in value.values():
            _ensure_finite(child, path, line)
    elif isinstance(value, list):
        for child in value:
            _ensure_finite(child, path, line)


def _loads_json(contents: str, path: Path, line: int = 1) -> Any:
    try:
        value = json.loads(contents, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        error_line = line + error.lineno - 1
        raise _validation_error(path, error_line, error.msg) from error
    except ValueError as error:
        raise _validation_error(path, line, str(error)) from error
    _ensure_finite(value, path, line)
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        contents = path.read_text()
    except FileNotFoundError as error:
        raise _validation_error(path, 1, "required file is missing") from error
    except OSError as error:
        raise _validation_error(path, 1, str(error)) from error
    value = _loads_json(contents, path)
    if not isinstance(value, dict):
        raise _validation_error(path, 1, "expected a JSON object")
    return value


def _read_metrics(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        stream = path.open()
    except FileNotFoundError as error:
        raise _validation_error(path, 1, "required file is missing") from error
    except OSError as error:
        raise _validation_error(path, 1, str(error)) from error

    metrics: list[dict[str, Any]] = []
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event = _loads_json(line, path, line_number)
            if not isinstance(event, dict):
                raise _validation_error(path, line_number, "expected a JSON object")
            step = event.get("step")
            if isinstance(step, bool) or not isinstance(step, int):
                raise _validation_error(path, line_number, "event step must be an integer")
            epoch = event.get("epoch")
            if isinstance(epoch, bool) or not isinstance(epoch, int):
                raise _validation_error(path, line_number, "event epoch must be an integer")
            event_type = event.get("type")
            if not isinstance(event_type, str) or not event_type:
                raise _validation_error(path, line_number, "event type must be a non-empty string")
            source_time = event.get("time")
            if not isinstance(source_time, str) or not source_time:
                raise _validation_error(path, line_number, "event time must be a non-empty string")
            try:
                datetime.fromisoformat(source_time.replace("Z", "+00:00"))
            except ValueError as error:
                raise _validation_error(path, line_number, "event time must be ISO 8601") from error
            metrics.append(event)
    return tuple(metrics)


def _safe_run_id(experiment: str, run_name: str) -> str:
    source = f"{experiment}/{run_name}"
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    prefix = _RUN_ID_PATTERN.sub("-", f"{experiment}-{run_name}").strip("-_") or "run"
    return f"{prefix[:80]}-{digest}"


def _run_directories(runs_directory: Path) -> list[Path]:
    if not runs_directory.is_dir():
        raise _validation_error(runs_directory, 1, "runs directory does not exist")
    directories: list[Path] = []
    try:
        experiments = sorted(path for path in runs_directory.iterdir() if path.is_dir())
        for experiment in experiments:
            directories.extend(sorted(path for path in experiment.iterdir() if path.is_dir()))
    except OSError as error:
        raise _validation_error(runs_directory, 1, str(error)) from error
    return directories


def validate_runs(runs_directory: str | Path) -> tuple[LocalRun, ...]:
    root = Path(runs_directory).resolve()
    local_runs: list[LocalRun] = []
    for directory in _run_directories(root):
        config_path = directory / "config.json"
        raw_config = _read_json_object(config_path)
        try:
            config = ExperimentConfig.model_validate(raw_config)
        except ValidationError as error:
            raise _validation_error(
                config_path, 1, f"invalid experiment config: {error}"
            ) from error
        environment = _read_json_object(directory / "environment.json")
        already_imported = (directory / "wandb.json").exists()
        metrics = () if already_imported else _read_metrics(directory / "metrics.jsonl")
        experiment = config.run.experiment
        if directory.parent.name != experiment:
            raise _validation_error(
                config_path,
                1,
                "config run.experiment does not match its parent directory",
            )
        local_runs.append(
            LocalRun(
                directory=directory,
                config=raw_config,
                environment=environment,
                metrics=metrics,
                experiment=experiment,
                display_name=wandb_run_name(config),
                tags=tuple(config.run.tags),
                max_steps=config.training.max_steps,
                run_id=_safe_run_id(experiment, directory.name),
                already_imported=already_imported,
            )
        )
    return tuple(local_runs)


def _source_status(local_run: LocalRun) -> str:
    run_end = next(
        (event for event in reversed(local_run.metrics) if event["type"] == "run_end"),
        None,
    )
    if run_end is None:
        return "interrupted"
    if run_end.get("completed") is True:
        return "completed"
    final_step = max((event["step"] for event in local_run.metrics), default=None)
    if local_run.max_steps is not None and final_step == local_run.max_steps:
        return "max_steps_reached"
    return "stopped"


def _wandb_tags(local_run: LocalRun) -> list[str]:
    additions = ["migrated"]
    if local_run.experiment.endswith("-pilot"):
        additions.append("pilot")
    return list(dict.fromkeys((*local_run.tags, *additions)))


def _wandb_config(local_run: LocalRun) -> dict[str, Any]:
    config = dict(local_run.config)
    config["environment"] = {
        key: value for key, value in local_run.environment.items() if key in _ENVIRONMENT_KEYS
    }
    return config


def _wandb_event(event: dict[str, Any]) -> dict[str, Any]:
    return dict(event)


def _summary(local_run: LocalRun) -> dict[str, Any]:
    return {
        "migration/status": _source_status(local_run),
        "migration/event_count": len(local_run.metrics),
        "migration/source_last_step": max(
            (event["step"] for event in local_run.metrics), default=None
        ),
        "migration/source_run": f"{local_run.directory.parent.name}/{local_run.directory.name}",
    }


def _write_marker(
    local_run: LocalRun,
    *,
    run_id: str,
    project: str,
    entity: str | None,
    url: str | None,
) -> None:
    marker = {
        "entity": entity,
        "id": run_id,
        "imported_at": datetime.now(UTC).isoformat(),
        "project": project,
        "url": url,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=local_run.directory,
        prefix=".wandb.json.",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
        temporary.replace(local_run.marker_path)
    finally:
        temporary.unlink(missing_ok=True)


def import_runs(
    runs_directory: str | Path,
    *,
    project: str,
    entity: str | None = None,
    dry_run: bool = False,
) -> ImportResult:
    local_runs = validate_runs(runs_directory)
    pending = tuple(run for run in local_runs if not run.already_imported)
    skipped = len(local_runs) - len(pending)
    events = sum(len(run.metrics) for run in pending)

    if dry_run:
        result = ImportResult(len(local_runs), 0, skipped, events)
        print(
            "wandb-import dry-run: "
            f"discovered={result.discovered} pending={len(pending)} "
            f"skipped={result.skipped} events={result.events}"
        )
        return result

    if not pending:
        result = ImportResult(len(local_runs), 0, skipped, 0)
        print(
            "wandb-import: "
            f"discovered={result.discovered} imported=0 skipped={result.skipped} events=0"
        )
        return result

    wandb = importlib.import_module("wandb")
    imported = 0
    for index, local_run in enumerate(pending, 1):
        print(
            f"wandb-import [{index}/{len(pending)}] "
            f"{local_run.experiment}/{local_run.directory.name}",
            flush=True,
        )
        initialization: dict[str, Any] = {
            "project": project,
            "id": local_run.run_id,
            "name": local_run.display_name,
            "group": local_run.experiment,
            "tags": _wandb_tags(local_run),
            "config": _wandb_config(local_run),
            "mode": "online",
            "force": True,
            "resume": "never",
        }
        if entity is not None:
            initialization["entity"] = entity
        remote_run = wandb.init(**initialization)
        if remote_run is None:
            raise RuntimeError("wandb.init did not return a run")
        try:
            remote_run.define_metric("step")
            remote_run.define_metric("*", step_metric="step", step_sync=False)
            for event in local_run.metrics:
                remote_run.log(_wandb_event(event))
            remote_run.summary.update(_summary(local_run))
        except BaseException:
            with suppress(Exception):
                remote_run.finish(exit_code=1)
            raise
        remote_id = getattr(remote_run, "id", None) or local_run.run_id
        remote_project = getattr(remote_run, "project", None) or project
        remote_entity = getattr(remote_run, "entity", None) or entity
        url = getattr(remote_run, "url", None)
        remote_run.finish(exit_code=0)
        _write_marker(
            local_run,
            run_id=remote_id,
            project=remote_project,
            entity=remote_entity,
            url=url,
        )
        imported += 1

    result = ImportResult(len(local_runs), imported, skipped, events)
    print(
        "wandb-import: "
        f"discovered={result.discovered} imported={result.imported} "
        f"skipped={result.skipped} events={result.events}"
    )
    return result
