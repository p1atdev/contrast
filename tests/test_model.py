import torch

from contrast.config.schema import ModelConfig, ProjectionConfig
from contrast.models import build_model


def test_vit_output_contract() -> None:
    config = ModelConfig(
        dim=48,
        depth=2,
        num_heads=3,
        projection=ProjectionConfig(hidden_dim=64, output_dim=16),
    )
    output = build_model(config)(torch.randn(3, 3, 32, 32))
    assert output.features.shape == (3, 48)
    assert output.embeddings.shape == (3, 16)
    assert output.raw_embeddings is not None
    assert output.raw_embeddings.shape == (3, 16)
    assert output.logits.shape == (3, 100)
    torch.testing.assert_close(output.embeddings.norm(dim=1), torch.ones(3))
