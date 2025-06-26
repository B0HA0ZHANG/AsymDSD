import torch
from torch import nn

from .activation import ActivationLayer
from .multilayer_perceptron import MLPVarLen
from .normalization import NormalizationLayer
from .tokenization import TrainableToken


class ClassificationHeadAttn(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 384,
        hidden_dims: tuple[int, ...] = (256, 256),
        num_heads: int = 1,
        norm_layer: NormalizationLayer | None = None,
        act_layer: ActivationLayer = nn.GELU,
        dropout_p: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.cls_token = TrainableToken(embed_dim=embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            dropout=dropout_p,
            bias=bias,
            batch_first=True,
        )

        self.mlp = MLPVarLen(
            *([embed_dim] + list(hidden_dims) + [num_classes]),
            norm_layer=norm_layer,
            act_layer=act_layer,
            dropout_p=dropout_p,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, 1, -1)

        # Only attention from cls token
        x, _ = self.attn(cls_token, x, x)
        x = self.mlp(x)

        return x
