from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .activation import ActivationLayer
from .multilayer_perceptron import MLPVarLen
from .normalization import NormalizationLayer


@dataclass
class ProjectionHeadConfig:
    in_dim: int = 384
    out_dim: int = 4096
    num_layers: int = 3
    hidden_dim: int = 1024
    bottleneck_dim: int = 256
    norm_layer: NormalizationLayer | None = None
    act_layer: ActivationLayer = nn.GELU
    bias: bool = True


class ProjectionOutput(NamedTuple):
    x: torch.Tensor
    x_norm: torch.Tensor | None = None


class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 384,
        out_dim: int = 4096,
        num_layers: int = 3,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 256,
        norm_layer: NormalizationLayer | None = None,
        act_layer: ActivationLayer = nn.GELU,
        bias: bool = True,
    ) -> None:
        super().__init__()
        dims = [in_dim]
        dims += [hidden_dim] * max((num_layers - 1), 0)
        dims += [bottleneck_dim]

        self.mlp = MLPVarLen(
            *dims, norm_layer=norm_layer, act_layer=act_layer, bias=bias
        )

        self.norm = partial(nn.functional.normalize, dim=-1, eps=1e-6)

        self.linear = nn.Linear(bottleneck_dim, out_dim, bias=bias)
        # Reparameterization with direction and magnitude
        self.linear = weight_norm(self.linear)

    def forward(self, x: torch.Tensor, return_x_norm=False) -> ProjectionOutput:
        x = self.mlp(x)
        norm = self.norm(x)
        x = self.linear(norm)
        return ProjectionOutput(x, norm if return_x_norm else None)
