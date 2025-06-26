import torch
from torch import nn


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_scale: float = 1e-1) -> None:
        super().__init__()
        self.scale = nn.Parameter(init_scale * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multiplication by diagonal matrix
        return self.scale * x
