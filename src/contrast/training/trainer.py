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


def _gradient_norm(parameters: tuple[nn.Parameter, ...]) -> torch.Tensor:
    gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.zeros(())
    norms = torch.stack([torch.linalg.vector_norm(gradient, ord=2) for gradient in gradients])
    return torch.linalg.vector_norm(norms, ord=2)


def _is_better(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "max":
        return value > best
    if mode == "min":
        return value < best
    raise ValueError(f"unsupported selection mode: {mode}")


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
        self.objective.initialize(self.model, total_steps=steps)
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
        self.best_metric_value: float | None = None
        self.best_metric_epoch: int | None = None

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
        selection = state.get("selection") or {}
        self.best_metric_value = selection.get("best_metric_value")
        self.best_metric_epoch = selection.get("best_metric_epoch")

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
                    "selection": {
                        "metric": self.config.evaluation.selection_metric,
                        "mode": self.config.evaluation.selection_mode,
                        "best_metric_value": self.best_metric_value,
                        "best_metric_epoch": self.best_metric_epoch,
                    },
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
            model_parameters = tuple(self.model.parameters())
            objective_parameters = tuple(self.objective.parameters())
            parameters = (*model_parameters, *objective_parameters)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                self.config.training.gradient_clip_norm or float("inf"),
                error_if_nonfinite=True,
            )
            should_log = (self.global_step + 1) % self.config.training.log_every_steps == 0
            gradient_diagnostics: tuple[float, float, float, float] | None = None
            if should_log:
                gradient_norm_value = float(gradient_norm)
                clip_norm = self.config.training.gradient_clip_norm
                clip_coefficient = (
                    min(1.0, clip_norm / (gradient_norm_value + 1e-6))
                    if clip_norm is not None
                    else 1.0
                )
                model_gradient_norm = float(_gradient_norm(model_parameters)) / max(
                    clip_coefficient, 1e-12
                )
                objective_gradient_norm = float(_gradient_norm(objective_parameters)) / max(
                    clip_coefficient, 1e-12
                )
                gradient_diagnostics = (
                    gradient_norm_value,
                    clip_coefficient,
                    model_gradient_norm,
                    objective_gradient_norm,
                )
            scaler = self.precision.scaler if self.precision.uses_scaler else None
            self.optimization.step(scaler)
            self.global_step += 1
            result.metrics.update(self.objective.after_optimizer_step(self.model, self.global_step))
            ema_updated = (
                self.ema.update(self.model, self.global_step) if self.ema is not None else False
            )

            if should_log:
                elapsed = time.perf_counter() - started
                assert gradient_diagnostics is not None
                (
                    gradient_norm_value,
                    clip_coefficient,
                    model_gradient_norm,
                    objective_gradient_norm,
                ) = gradient_diagnostics
                metrics: dict[str, Any] = {
                    "type": "train",
                    "epoch": self.epoch,
                    "step": self.global_step,
                    "step_in_epoch": step_in_epoch,
                    "loss": float(result.loss.detach()),
                    "optimization/lr": self.optimization.lr,
                    "optimization/gradient_norm": gradient_norm_value,
                    "optimization/model_gradient_norm": model_gradient_norm,
                    "optimization/objective_gradient_norm": objective_gradient_norm,
                    "optimization/gradient_clip_coefficient": clip_coefficient,
                    "optimization/clipped_gradient_norm": (gradient_norm_value * clip_coefficient),
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
                if self.runtime.device.type == "cuda":
                    device = self.runtime.device
                    mebibyte = 1024**2
                    metrics.update(
                        {
                            "performance/cuda_memory_allocated_mib": (
                                torch.cuda.memory_allocated(device) / mebibyte
                            ),
                            "performance/cuda_memory_reserved_mib": (
                                torch.cuda.memory_reserved(device) / mebibyte
                            ),
                            "performance/cuda_peak_memory_allocated_mib": (
                                torch.cuda.max_memory_allocated(device) / mebibyte
                            ),
                            "performance/cuda_peak_memory_reserved_mib": (
                                torch.cuda.max_memory_reserved(device) / mebibyte
                            ),
                        }
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
        selection_metric = self.config.evaluation.selection_metric
        is_best = False
        if selection_metric is not None:
            if selection_metric not in metrics:
                raise KeyError(f"evaluation selection metric was not produced: {selection_metric}")
            selection_value = metrics[selection_metric]
            is_best = _is_better(
                selection_value,
                self.best_metric_value,
                self.config.evaluation.selection_mode,
            )
            if is_best:
                self.best_metric_value = selection_value
                self.best_metric_epoch = self.epoch
            metrics.update(
                {
                    "selection/value": selection_value,
                    "selection/best_value": self.best_metric_value,
                    "selection/best_epoch": float(self.best_metric_epoch),
                    "selection/is_best": float(is_best),
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
        if is_best and self.config.evaluation.save_best_checkpoint:
            self._checkpoint("best.pt")

    def fit(self) -> Path:
        completed = True
        stopped_at_max_steps = False
        for epoch in range(self.epoch, self.config.training.epochs):
            self.epoch = epoch
            epoch_completed = self._train_epoch()
            stopped_at_max_steps = bool(
                not epoch_completed
                and self.config.training.max_steps is not None
                and self.global_step >= self.config.training.max_steps
            )
            completed = epoch_completed or stopped_at_max_steps
            is_final_epoch = epoch_completed and epoch + 1 == self.config.training.epochs
            if (
                epoch_completed
                and self.config.evaluation.enabled
                and (
                    (epoch + 1) % self.config.training.evaluate_every_epochs == 0 or is_final_epoch
                )
            ):
                self._evaluate(final=is_final_epoch)
            if (
                epoch_completed
                and not is_final_epoch
                and (epoch + 1) % self.config.training.checkpoint_every_epochs == 0
            ):
                self._checkpoint(f"epoch-{epoch + 1:04d}.pt")
            if not epoch_completed:
                break
        final = self._checkpoint("final.pt")
        self.store.log(
            {
                "type": "run_end",
                "epoch": self.epoch,
                "step": self.global_step,
                "completed": completed,
                "stopped_at_max_steps": stopped_at_max_steps,
            }
        )
        return final
