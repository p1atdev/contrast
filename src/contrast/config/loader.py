from __future__ import annotations

import json
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

from contrast.config.schema import ExperimentConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            if "kind" in value and value["kind"] != merged[key].get("kind"):
                # A discriminator change selects a different schema; stale fields are invalid.
                merged[key] = deepcopy(value)
            else:
                merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_toml_tree(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"cyclic config extends: {chain}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    extends = raw.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    merged: dict[str, Any] = {}
    for parent in extends:
        merged = _deep_merge(merged, _load_toml_tree(path.parent / parent, (*stack, path)))
    return _deep_merge(merged, raw)


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _apply_override(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"override must be KEY=VALUE, received: {expression}")
    dotted_key, raw_value = expression.split("=", 1)
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        child = target.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot assign through non-table key: {dotted_key}")
        target = child
    target[keys[-1]] = _parse_override_value(raw_value)


def load_experiment_config(
    path: str | Path,
    overrides: list[str] | tuple[str, ...] = (),
) -> ExperimentConfig:
    raw = _load_toml_tree(Path(path))
    for expression in overrides:
        _apply_override(raw, expression)
    return ExperimentConfig.model_validate(raw)


def canonical_config_json(config: ExperimentConfig) -> str:
    return json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
