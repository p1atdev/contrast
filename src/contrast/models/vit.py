from __future__ import annotations

import torch
from torch import nn

from contrast.models.layers import TransformerBlock, build_activation, build_norm


class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        normalization: str,
        activation: str,
        attention_backend: str,
        norm_eps: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image size must be divisible by patch size")
        patch_count = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, patch_count + 1, dim))
        self.input_dropout = nn.Dropout(dropout)

        def norm_factory() -> nn.Module:
            return build_norm(normalization, dim, norm_eps)

        def activation_factory() -> nn.Module:
            return build_activation(activation)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_factory=norm_factory,
                    activation_factory=activation_factory,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    attention_backend=attention_backend,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = norm_factory()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = self.input_dropout(torch.cat((cls, patches), dim=1) + self.position_embedding)
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens)[:, 0]
