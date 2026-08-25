from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class RuntimeContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @classmethod
    def initialize(cls) -> RuntimeContext:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        distributed = world_size > 1

        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank if distributed else 0)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        if distributed and not dist.is_initialized():
            backend = "nccl" if device.type == "cuda" else "gloo"
            dist.init_process_group(backend=backend)
        return cls(rank, local_rank, world_size, device, distributed)

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def mean(self, value: torch.Tensor) -> torch.Tensor:
        if not self.distributed:
            return value
        result = value.detach().clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result / self.world_size

    def close(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()
