import torch
from torch import Tensor, nn


class TransposeBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input.transpose(1, 2)).transpose(1, 2)


class RMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | list[int],
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if isinstance(normalized_shape, int):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore
        self.normalized_shape = tuple(normalized_shape)  # type: ignore

        self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.eps = eps

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


NormalizationLayer = (
    type[nn.LayerNorm]
    | type[RMSNorm]
    | type[nn.BatchNorm1d]
    | type[TransposeBatchNorm1d]
    | type[nn.Identity]
)
