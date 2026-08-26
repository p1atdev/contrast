from __future__ import annotations

import argparse
import copy
import itertools
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError

from contrast.config.loader import canonical_config_json, load_experiment_config
from contrast.config.schema import ExperimentConfig
from contrast.data import build_cifar100_loaders
from contrast.models import build_model
from contrast.objectives import build_objective
from contrast.reproducibility import seed_everything
from contrast.runtime import PrecisionManager, RuntimeContext
from contrast.tracking import RunStore
from contrast.training import Trainer
from contrast.training.ema import ExponentialMovingAverage
from contrast.training.evaluation import evaluate


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


def _config_from_checkpoint(state: dict[str, Any]) -> ExperimentConfig:
    raw_config = copy.deepcopy(state["config"])
    data_config = raw_config.setdefault("data", {})
    if "split_seed" not in data_config:
        data_config["split_seed"] = raw_config["run"]["seed"]
    return ExperimentConfig.model_validate(raw_config)


def _evaluate_checkpoint(arguments: argparse.Namespace) -> int:
    checkpoint = Path(arguments.checkpoint).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _config_from_checkpoint(state)
    run_directory = (
        Path(arguments.run_dir).resolve() if arguments.run_dir else checkpoint.parent.parent
    )
    store = RunStore.for_existing_run(run_directory)
    runtime = RuntimeContext.initialize()
    try:
        if runtime.distributed:
            raise NotImplementedError("offline evaluation supports a single process")
        seed_everything(config.run.seed)
        precision = PrecisionManager(config.precision, runtime.device)
        precision.configure_backends(config.reproducibility)
        data = build_cifar100_loaders(config, runtime)
        model = build_model(config.model).to(runtime.device)
        model.load_state_dict(state["model"])
        ema = None
        if config.ema.enabled:
            if state.get("ema") is None:
                raise ValueError("checkpoint config enables EMA but contains no EMA state")
            ema = ExponentialMovingAverage(model, config.ema)
            ema.load_state_dict(state["ema"])
        query_loaders = {"eval": data.validation or data.test}
        if config.evaluation.test_at_end:
            query_loaders["test"] = data.test
        evaluation_weights = config.ema.evaluation_weights if ema is not None else "raw"
        metrics: dict[str, float] = {}
        if evaluation_weights in {"raw", "both"}:
            metrics.update(
                evaluate(
                    model,
                    data.memory,
                    query_loaders,
                    runtime.device,
                    precision,
                    config.evaluation,
                    include_linear_probe=True,
                )
            )
        if ema is not None and ema.updates and evaluation_weights in {"ema", "both"}:
            ema_query_loaders = {f"{name}_ema": loader for name, loader in query_loaders.items()}
            metrics.update(
                evaluate(
                    ema.model,
                    data.memory,
                    ema_query_loaders,
                    runtime.device,
                    precision,
                    config.evaluation,
                    include_linear_probe=True,
                )
            )
        if ema is not None:
            metrics.update(
                {
                    "ema/decay": ema.decay,
                    "ema/updates": float(ema.updates),
                    "ema/ready": float(ema.updates > 0),
                }
            )
        store.log(
            {
                "type": "offline_evaluation",
                "epoch": int(state.get("epoch", 0)),
                "step": int(state.get("global_step", 0)),
                "checkpoint": str(checkpoint),
                **metrics,
            }
        )
        if runtime.is_primary:
            print(f"run={store.directory}")
            print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    finally:
        runtime.close()


def _serialize_override(key: str, value: Any) -> str:
    return f"{key}={json.dumps(value, separators=(',', ':'))}"


def _expand_sweep(path: Path) -> list[tuple[Path, list[str]]]:
    with path.open("rb") as stream:
        specification = tomllib.load(stream)

    config_values = specification.get("configs")
    if config_values is None:
        base = specification.get("base")
        config_values = [base] if base is not None else []
    if (
        not isinstance(config_values, list)
        or not config_values
        or not all(isinstance(value, str) for value in config_values)
    ):
        raise ValueError("sweep requires a non-empty string 'base' or 'configs' list")

    grid = specification.get("grid")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("sweep requires a non-empty 'grid' table")
    if any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError("every sweep grid value must be a non-empty array")

    keys = list(grid)
    runs: list[tuple[Path, list[str]]] = []
    for values in itertools.product(*(grid[key] for key in keys)):
        overrides = [
            _serialize_override(key, value) for key, value in zip(keys, values, strict=True)
        ]
        for config_value in config_values:
            config_path = (path.parent / config_value).resolve()
            load_experiment_config(config_path, overrides)
            runs.append((config_path, overrides))
    return runs


def _sweep(arguments: argparse.Namespace) -> int:
    path = Path(arguments.sweep).resolve()
    runs = _expand_sweep(path)
    print(f"sweep combinations={len(runs)}")
    for index, (config_path, overrides) in enumerate(runs, 1):
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
            f"[{index}/{len(runs)}] {config_path.name} {' '.join(overrides)}",
            flush=True,
        )
        if not arguments.dry_run:
            subprocess.run(command, check=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contrast")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="run one experiment")
    train.add_argument("--config", "-c", required=True)
    train.add_argument("--set", dest="overrides", action="append", default=[])
    train.add_argument("--resume")
    train.set_defaults(handler=_train)

    offline = commands.add_parser("evaluate", help="evaluate a saved checkpoint")
    offline.add_argument("--checkpoint", required=True)
    offline.add_argument("--run-dir", help="run receiving the appended metrics")
    offline.set_defaults(handler=_evaluate_checkpoint)

    validate = commands.add_parser("validate", help="resolve and validate a config")
    validate.add_argument("--config", "-c", required=True)
    validate.add_argument("--set", dest="overrides", action="append", default=[])
    validate.set_defaults(handler=_validate)

    sweep = commands.add_parser("sweep", help="run a Cartesian TOML sweep sequentially")
    sweep.add_argument("sweep")
    sweep.add_argument("--dry-run", action="store_true")
    sweep.set_defaults(handler=_sweep)

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
