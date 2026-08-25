from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np
import torch


def derive_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join([str(base_seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass
class RngState:
    python: object
    numpy: tuple[object, ...]
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor]


def capture_rng_state() -> RngState:
    return RngState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng_state(state: RngState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    if state.torch_cuda and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.torch_cuda)
