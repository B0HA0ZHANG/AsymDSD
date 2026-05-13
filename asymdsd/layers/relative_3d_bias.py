from dataclasses import dataclass

import torch
from torch import nn

from ..components import FactoryConfig


@dataclass
class Relative3DBiasConfig(FactoryConfig):
    num_heads: int | None = None
    hidden_dim: int = 64
    use_distance: bool = True
    rbf_num_bins: int = 0
    rbf_max_distance: float = 2.0
    bias_bound: float | None = None

    @property
    def CLS(self):
        return Relative3DBias

    def instantiate(self) -> "Relative3DBias":
        if self.num_heads is None:
            raise ValueError("num_heads must be specified for Relative3DBias.")

        return self.CLS(
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            use_distance=self.use_distance,
            rbf_num_bins=self.rbf_num_bins,
            rbf_max_distance=self.rbf_max_distance,
            bias_bound=self.bias_bound,
        )


class Relative3DBias(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 64,
        use_distance: bool = True,
        rbf_num_bins: int = 0,
        rbf_max_distance: float = 2.0,
        bias_bound: float | None = None,
    ) -> None:
        super().__init__()
        if rbf_num_bins < 0:
            raise ValueError(f"rbf_num_bins must be >= 0, got {rbf_num_bins}.")
        if rbf_num_bins > 0 and rbf_max_distance <= 0.0:
            raise ValueError(
                "rbf_max_distance must be > 0 when using RBF bins, "
                f"got {rbf_max_distance}."
            )
        if bias_bound is not None and bias_bound <= 0.0:
            raise ValueError(f"bias_bound must be > 0, got {bias_bound}.")

        self.num_heads = num_heads
        self.use_distance = use_distance
        self.rbf_num_bins = rbf_num_bins
        self.bias_bound = bias_bound

        if rbf_num_bins > 0:
            centers = torch.linspace(0.0, rbf_max_distance, rbf_num_bins)
            width = (
                rbf_max_distance
                if rbf_num_bins == 1
                else rbf_max_distance / (rbf_num_bins - 1)
            )
            self.register_buffer("rbf_centers", centers)
            self.register_buffer("rbf_gamma", torch.tensor(1.0 / (width * width)))

        in_dim = 3
        if use_distance:
            in_dim += 1
        in_dim += rbf_num_bins
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads, bias=False),
        )

    def forward(
        self,
        q_centers: torch.Tensor,
        k_centers: torch.Tensor,
    ) -> torch.Tensor:
        rel = q_centers.unsqueeze(2) - k_centers.unsqueeze(1)
        feats = [rel]

        if self.use_distance or self.rbf_num_bins > 0:
            dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
            if self.use_distance:
                feats.append(dist)
            if self.rbf_num_bins > 0:
                centers = self.rbf_centers.to(dtype=dist.dtype)
                gamma = self.rbf_gamma.to(dtype=dist.dtype)
                feats.append(torch.exp(-gamma * (dist - centers).square()))

        bias = self.mlp(torch.cat(feats, dim=-1))
        if self.bias_bound is not None:
            bias = self.bias_bound * torch.tanh(bias / self.bias_bound)
        return bias.permute(0, 3, 1, 2).contiguous()
