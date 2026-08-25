from __future__ import annotations

import torch


def valid_pair_mask(size: int, device: torch.device) -> torch.Tensor:
    return ~torch.eye(size, dtype=torch.bool, device=device)


def class_positive_mask(labels: torch.Tensor) -> torch.Tensor:
    valid = valid_pair_mask(labels.numel(), labels.device)
    return labels[:, None].eq(labels[None, :]) & valid


def instance_positive_mask(source_ids: torch.Tensor) -> torch.Tensor:
    valid = valid_pair_mask(source_ids.numel(), source_ids.device)
    return source_ids[:, None].eq(source_ids[None, :]) & valid
