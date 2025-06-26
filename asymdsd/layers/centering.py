import torch
import torch.distributed as dist
from torch import nn


class Centering(nn.Module):
    def __init__(self, dim: int, power_law_tau: float | None = None) -> None:
        super().__init__()
        self.register_buffer("center", torch.zeros(dim))
        self.power_law_tau = power_law_tau

        if power_law_tau is not None:
            indices = torch.arange(1, dim + 1)
            power_law_logits = power_law_tau * torch.log(indices.float())
            # Center the power law logits
            power_law_logits -= power_law_logits.mean()

            self.register_buffer("power_law_logits", power_law_logits)
            self.power_law_logits: torch.Tensor

    @torch.no_grad()
    def update_center(self, x: torch.Tensor, momentum: float) -> None:
        # Take mean over batch and patch dimension
        batch_mean = x.mean(dim=(0, 1))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_mean, op=dist.ReduceOp.AVG)

        self.center = momentum * self.center + (1 - momentum) * batch_mean

    def forward(self, x: torch.Tensor, *, momentum: float) -> torch.Tensor:
        x_centered = x - self.center

        # TODO: Add option to compute center from different input.
        if self.power_law_tau is not None:
            sort_idx = torch.argsort(self.center)
            x_centered[..., sort_idx] += self.power_law_logits

        self.update_center(x, momentum)
        return x_centered
