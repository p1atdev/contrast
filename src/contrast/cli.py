from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from contrast.config.loader import canonical_config_json, load_experiment_config
from contrast.data import build_cifar100_loaders
from contrast.models import build_model
from contrast.objectives import build_objective
from contrast.reproducibility import seed_everything
from contrast.runtime import PrecisionManager, RuntimeContext
from contrast.tracking import RunStore
from contrast.training import Trainer


def _train(arguments: argparse.Namespace) -> int:
    config = load_experiment_config(arguments.config, arguments.overrides)
    runtime = RuntimeContext.initialize()
    try:
        if runtime.distributed:
            raise NotImplementedError(
                "distributed execution is reserved by the runtime interface but global "
                "representation gathering is not implemented yet"
            )
        seed_everything(config.run.seed)
        precision = PrecisionManager(config.precision, runtime.device)
        precision.configure_backends(config.reproducibility)
        data = build_cifar100_loaders(config, runtime)
        store = RunStore(config)
        trainer = Trainer(
            config,
            runtime,
            precision,
            build_model(config.model),
            build_objective(config.objective),
            data,
            store,
        )
        if arguments.resume:
            trainer.load_checkpoint(arguments.resume)
        final = trainer.fit()
        if runtime.is_primary:
            print(f"run={store.directory}\ncheckpoint={final}")
        return 0
    finally:
        runtime.close()


def _validate(arguments: argparse.Namespace) -> int:
    config = load_experiment_config(arguments.config, arguments.overrides)
    print(canonical_config_json(config), end="")
    return 0


def _serialize_override(key: str, value: Any) -> str:
    return f"{key}={json.dumps(value, separators=(',', ':'))}"


def _sweep(arguments: argparse.Namespace) -> int:
    path = Path(arguments.sweep).resolve()
    with path.open("rb") as stream:
        specification = tomllib.load(stream)
    config_paths = specification.get("configs", [specification.get("base")])
    if not config_paths or config_paths == [None]:
        raise ValueError("sweep requires either 'base' or 'configs'")
    grid: dict[str, list[Any]] = specification["grid"]
    keys = list(grid)
    combinations = list(itertools.product(config_paths, *(grid[key] for key in keys)))
    print(f"sweep combinations={len(combinations)}")
    for index, values in enumerate(combinations, 1):
        config_path = (path.parent / values[0]).resolve()
        overrides = [
            _serialize_override(key, value) for key, value in zip(keys, values[1:], strict=True)
        ]
        command = [
            sys.executable,
            "-m",
            "contrast.cli",
            "train",
            "--config",
            str(config_path),
            *itertools.chain.from_iterable(("--set", value) for value in overrides),
        ]
        print(
            f"[{index}/{len(combinations)}] {config_path.name} {' '.join(overrides)}",
            flush=True,
        )
        if not arguments.dry_run:
            subprocess.run(command, check=True)
    return 0


def _serve(arguments: argparse.Namespace) -> int:
    import uvicorn

    from contrast.web.app import create_app

    uvicorn.run(
        create_app(Path(arguments.runs_dir)),
        host=arguments.host,
        port=arguments.port,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contrast")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="run one experiment")
    train.add_argument("--config", "-c", required=True)
    train.add_argument("--set", dest="overrides", action="append", default=[])
    train.add_argument("--resume")
    train.set_defaults(handler=_train)

    validate = commands.add_parser("validate", help="resolve and validate a config")
    validate.add_argument("--config", "-c", required=True)
    validate.add_argument("--set", dest="overrides", action="append", default=[])
    validate.set_defaults(handler=_validate)

    sweep = commands.add_parser("sweep", help="run a Cartesian TOML sweep sequentially")
    sweep.add_argument("sweep")
    sweep.add_argument("--dry-run", action="store_true")
    sweep.set_defaults(handler=_sweep)

    serve = commands.add_parser("serve", help="serve the local run dashboard")
    serve.add_argument("--runs-dir", default="runs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_serve)
    return parser


def main() -> None:
    try:
        arguments = build_parser().parse_args()
        raise SystemExit(arguments.handler(arguments))
    except ValidationError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
