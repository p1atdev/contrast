from __future__ import annotations

import time
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from torch import nn

from contrast.config.schema import ExperimentConfig
from contrast.data.cifar import DataBundle
from contrast.objectives.base import Objective
from contrast.reproducibility import capture_rng_state, restore_rng_state
from contrast.runtime.context import RuntimeContext
from contrast.runtime.precision import PrecisionManager
from contrast.tracking.store import RunStore
from contrast.training.ema import ExponentialMovingAverage
from contrast.training.evaluation import evaluate
from contrast.training.optimization import OptimizationController
from contrast.training.steps import DirectStep, GradCacheStep, prepare_batch


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        runtime: RuntimeContext,
        precision: PrecisionManager,
        model: nn.Module,
        objective: Objective,
        data: DataBundle,
        store: RunStore,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.precision = precision
        self.model = model.to(runtime.device)
        self.objective = objective.to(runtime.device)
        self.data = data
        self.store = store
        steps = len(data.train) * config.training.epochs
        if config.training.max_steps is not None:
            steps = min(steps, config.training.max_steps)
        named_parameters = chain(
            ((f"model.{name}", parameter) for name, parameter in self.model.named_parameters()),
            (
                (f"objective.{name}", parameter)
                for name, parameter in self.objective.named_parameters()
            ),
        )
        self.optimization = OptimizationController(config.optimizer, named_parameters, steps)
        self.ema = ExponentialMovingAverage(self.model, config.ema) if config.ema.enabled else None
        self.strategy = (
            GradCacheStep(
                precision,
                config.batch.grad_cache_chunk_size_per_rank * config.batch.views,
            )
            if config.training.step_strategy == "grad_cache"
            else DirectStep(precision)
        )
        self.epoch = 0
        self.global_step = 0

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        if self.ema is not None:
            if state.get("ema") is None:
                self.ema.reset(self.model)
            else:
                self.ema.load_state_dict(state["ema"])
        self.objective.load_state_dict(state["objective"])
        self.optimization.load_state_dict(state["optimization"])
        restore_rng_state(state["rng"])
        self.epoch = int(state["epoch"]) + 1
        self.global_step = int(state["global_step"])

    def _checkpoint(self, name: str) -> Path:
        with self.optimization.evaluation_parameters():
            return self.store.save_checkpoint(
                {
                    "schema_version": 1,
                    "model": self.model.state_dict(),
                    "objective": self.objective.state_dict(),
                    "optimization": self.optimization.state_dict(),
                    "rng": capture_rng_state(),
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                    "config": self.config.model_dump(mode="json"),
                    "ema": self.ema.state_dict() if self.ema is not None else None,
                },
                name,
            )

    def _train_epoch(self) -> bool:
        self.model.train()
        self.objective.train()
        self.optimization.train()
        self.data.train_sampler.set_epoch(self.epoch)
        started = time.perf_counter()
        for step_in_epoch, raw_batch in enumerate(self.data.train):
            if (
                self.config.training.max_steps is not None
                and self.global_step >= self.config.training.max_steps
            ):
                return False
            prepared = prepare_batch(raw_batch, self.runtime.device)
            self.optimization.zero_grad()
            result = self.strategy.backward(self.model, self.objective, prepared)
            if self.precision.uses_scaler:
                self.precision.scaler.unscale_(self.optimization.optimizer)
            parameters = chain(self.model.parameters(), self.objective.parameters())
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                self.config.training.gradient_clip_norm or float("inf"),
                error_if_nonfinite=True,
            )
            scaler = self.precision.scaler if self.precision.uses_scaler else None
            self.optimization.step(scaler)
            self.global_step += 1
            ema_updated = (
                self.ema.update(self.model, self.global_step) if self.ema is not None else False
            )

            if self.global_step % self.config.training.log_every_steps == 0:
                elapsed = time.perf_counter() - started
                gradient_norm_value = float(gradient_norm)
                metrics: dict[str, Any] = {
                    "type": "train",
                    "epoch": self.epoch,
                    "step": self.global_step,
                    "step_in_epoch": step_in_epoch,
                    "loss": float(result.loss.detach()),
                    "optimization/lr": self.optimization.lr,
                    "optimization/gradient_norm": gradient_norm_value,
                    "performance/sources_per_second": (
                        (step_in_epoch + 1)
                        * self.config.batch.global_source_batch_size
                        / max(elapsed, 1e-9)
                    ),
                }
                if self.config.training.gradient_clip_norm is not None:
                    metrics["optimization/gradient_was_clipped"] = float(
                        gradient_norm_value > self.config.training.gradient_clip_norm
                    )
                if self.ema is not None:
                    metrics["ema/decay"] = self.ema.decay
                    metrics["ema/updates"] = self.ema.updates
                    metrics["ema/updated"] = float(ema_updated)
                metrics.update({key: float(value) for key, value in result.metrics.items()})
                self.store.log(metrics)
                if self.runtime.is_primary:
                    print(
                        f"epoch={self.epoch} step={self.global_step} "
                        f"loss={metrics['loss']:.5f} lr={self.optimization.lr:.3g}",
                        flush=True,
                    )
        return True

    def _evaluate(self, *, final: bool = False) -> None:
        query_loaders = {"eval": self.data.validation or self.data.test}
        if final and self.config.evaluation.test_at_end:
            query_loaders["test"] = self.data.test
        evaluation_weights = self.config.ema.evaluation_weights if self.ema is not None else "raw"
        metrics: dict[str, float] = {}
        with self.optimization.evaluation_parameters():
            if evaluation_weights in {"raw", "both"}:
                metrics.update(
                    evaluate(
                        self.model,
                        self.data.memory,
                        query_loaders,
                        self.runtime.device,
                        self.precision,
                        self.config.evaluation,
                        include_linear_probe=final,
                    )
                )
            if self.ema is not None and self.ema.updates and evaluation_weights in {"ema", "both"}:
                ema_query_loaders = {
                    f"{name}_ema": loader for name, loader in query_loaders.items()
                }
                metrics.update(
                    evaluate(
                        self.ema.model,
                        self.data.memory,
                        ema_query_loaders,
                        self.runtime.device,
                        self.precision,
                        self.config.evaluation,
                        include_linear_probe=final,
                    )
                )
        if self.ema is not None:
            metrics.update(
                {
                    "ema/decay": self.ema.decay,
                    "ema/updates": float(self.ema.updates),
                    "ema/ready": float(self.ema.updates > 0),
                }
            )
        self.store.log(
            {
                "type": "final_evaluation" if final else "evaluation",
                "epoch": self.epoch,
                "step": self.global_step,
                **metrics,
            }
        )
        if self.runtime.is_primary:
            summary = " ".join(
                f"{key}={value:.4f}" for key, value in metrics.items() if key.endswith("top1")
            )
            print(f"evaluation epoch={self.epoch} {summary}", flush=True)

    def fit(self) -> Path:
        completed = True
        for epoch in range(self.epoch, self.config.training.epochs):
            self.epoch = epoch
            completed = self._train_epoch()
            is_final_epoch = completed and epoch + 1 == self.config.training.epochs
            if self.config.evaluation.enabled and (
                (epoch + 1) % self.config.training.evaluate_every_epochs == 0 or is_final_epoch
            ):
                self._evaluate(final=is_final_epoch)
            if (epoch + 1) % self.config.training.checkpoint_every_epochs == 0:
                self._checkpoint(f"epoch-{epoch + 1:04d}.pt")
            if not completed:
                break
        final = self._checkpoint("final.pt")
        self.store.log(
            {
                "type": "run_end",
                "epoch": self.epoch,
                "step": self.global_step,
                "completed": completed,
            }
        )
        return final
