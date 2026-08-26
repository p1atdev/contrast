from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from contrast.config.loader import canonical_config_json
from contrast.config.schema import ExperimentConfig


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


class RunStore:
    def __init__(self, config: ExperimentConfig) -> None:
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
        self.metrics_path = self.directory / "metrics.jsonl"

    @classmethod
    def for_existing_run(cls, directory: str | Path) -> RunStore:
        resolved = Path(directory).resolve()
        if not (resolved / "config.json").is_file():
            raise ValueError(f"not a run directory: {resolved}")
        store = cls.__new__(cls)
        store.directory = resolved
        store.checkpoint_directory = resolved / "checkpoints"
        store.metrics_path = resolved / "metrics.jsonl"
        return store

    def log(self, values: dict[str, Any]) -> None:
        event = {"time": datetime.now(UTC).isoformat(), **values}
        with self.metrics_path.open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

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
