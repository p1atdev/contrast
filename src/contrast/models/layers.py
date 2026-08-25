from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = F.rms_norm(inputs, (inputs.shape[-1],), eps=self.eps)
        return normalized * self.weight


def build_norm(name: str, dim: int, eps: float) -> nn.Module:
    if name == "layer_norm":
        return nn.LayerNorm(dim, eps=eps)
    if name == "rms_norm":
        return RMSNorm(dim, eps=eps)
    raise ValueError(f"unknown normalization: {name}")


def build_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unknown activation: {name}")


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.backend = backend
        self.dropout = dropout
        self.qkv = nn.Linear(dim, dim * 3)
        self.output = nn.Linear(dim, dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        dropout = self.dropout if self.training else 0.0

        if self.backend == "eager":
            scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * self.scale
            weights = F.softmax(scores, dim=-1).to(query.dtype)
            weights = F.dropout(weights, p=dropout, training=self.training)
            attended = torch.matmul(weights, value)
        elif self.backend == "sdpa":
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=dropout,
            )
        else:
            raise ValueError(f"unknown attention backend: {self.backend}")

        attended = attended.transpose(1, 2).reshape(batch, tokens, dim)
        return self.output_dropout(self.output(attended))


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation_factory: Callable[[], nn.Module],
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation_factory(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        norm_factory: Callable[[], nn.Module],
        activation_factory: Callable[[], nn.Module],
        dropout: float,
        attention_dropout: float,
        attention_backend: str,
    ) -> None:
        super().__init__()
        self.attention_norm = norm_factory()
        self.attention = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            backend=attention_backend,
        )
        self.mlp_norm = norm_factory()
        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            activation_factory=activation_factory,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.mlp(self.mlp_norm(inputs))
