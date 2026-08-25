from contrast.config.schema import ModelConfig
from contrast.models.model import ContrastiveModel
from contrast.models.vit import VisionTransformer


def build_model(config: ModelConfig) -> ContrastiveModel:
    encoder = VisionTransformer(
        image_size=config.image_size,
        patch_size=config.patch_size,
        dim=config.dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        normalization=config.layers.normalization,
        activation=config.layers.activation,
        attention_backend=config.layers.attention,
        norm_eps=config.layers.norm_eps,
    )
    return ContrastiveModel(
        encoder=encoder,
        feature_dim=config.dim,
        projection_hidden_dim=config.projection.hidden_dim,
        embedding_dim=config.projection.output_dim,
        num_classes=config.num_classes,
    )
