from __future__ import annotations

from contrast.config.schema import ExperimentConfig


def wandb_run_name(config: ExperimentConfig) -> str:
    """Build a readable W&B name from the experiment, condition, and seed."""
    condition = config.objective.kind.replace("_", "-")
    return f"{config.run.experiment}/{condition}-seed-{config.run.seed}"
