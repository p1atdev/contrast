from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch

from contrast.config.schema import PrecisionConfig, ReproducibilityConfig


class PrecisionManager:
    def __init__(self, config: PrecisionConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.autocast_dtype = {
            "none": None,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[config.autocast_dtype]
        scaler_enabled = device.type == "cuda" and self.autocast_dtype is torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    def configure_backends(self, reproducibility: ReproducibilityConfig) -> None:
        torch.set_float32_matmul_precision(self.config.matmul_precision)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self.config.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.config.allow_tf32
            torch.backends.cudnn.benchmark = reproducibility.cudnn_benchmark
        torch.use_deterministic_algorithms(reproducibility.mode == "strict")

    def autocast(self) -> AbstractContextManager[None]:
        if self.device.type != "cuda" or self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    @property
    def uses_scaler(self) -> bool:
        return self.scaler.is_enabled()
