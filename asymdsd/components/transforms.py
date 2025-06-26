from abc import ABC, abstractmethod
from typing import Sequence

import torch
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import axis_angle_to_matrix, random_rotations
from torch import nn

from ..components.common_types import OptionalTensor
from .common_types import OneOrSequence_T
from .utils import xyz_view


class CenterPC(nn.Module):
    def forward(
        self, points: torch.Tensor, mask: OptionalTensor = None
    ) -> torch.Tensor:
        xyz = xyz_view(points)  # (B, N, 3)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            xyz_masked = xyz * mask
            center = xyz_masked.sum(dim=-2, keepdim=True) / mask.sum(
                dim=-2, keepdim=True
            )
        else:
            center = xyz.mean(dim=-2, keepdim=True)  # (B, 1, 3)

        xyz -= center

        points[..., :3] = xyz
        return points


class NormalizeUnitSpherePC(nn.Module):
    def __init__(self):
        super().__init__()
        self.center_pc = CenterPC()

    def forward(
        self, points: torch.Tensor, mask: OptionalTensor = None
    ) -> torch.Tensor:
        points = self.center_pc(points, mask)

        xyz = xyz_view(points)  # (B, N, 3)
        norm = torch.norm(xyz, dim=-1, keepdim=True)  # (B, N, 1)

        if mask is not None:
            norm_masked = norm * mask.unsqueeze(-1)
            scale = norm_masked.amax(dim=(-1, -2), keepdim=True)  # (B, 1, 1)
        else:
            scale = norm.amax(dim=(-1, -2), keepdim=True)  # (B, 1, 1)

        xyz /= scale

        points[..., :3] = xyz
        return points


class NormalizePC(nn.Module):
    def __init__(self):
        super().__init__()
        self.center_pc = CenterPC()

    def forward(
        self, points: torch.Tensor, mask: OptionalTensor = None
    ) -> torch.Tensor:
        points = self.center_pc(points, mask)
        xyz = xyz_view(points)

        if mask is not None:
            xyz_masked = xyz * mask.unsqueeze(-1)
            std = xyz_masked.std(dim=(-1, -2), keepdim=True)
        else:
            std = xyz.std(dim=(-1, -2), keepdim=True)

        xyz /= std

        points[..., :3] = xyz
        return points


class FarthestPointSubSamplePC(nn.Module):
    def __init__(self, num_points: int = 1024):
        super().__init__()
        self.num_points = num_points

    def forward(
        self, points: torch.Tensor, num_points: OptionalTensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = xyz_view(points)

        _, idx = sample_farthest_points(
            xyz, lengths=num_points, K=self.num_points, random_start_point=True
        )
        points = points.gather(
            dim=1, index=idx.unsqueeze(-1).expand(-1, -1, points.shape[-1])
        )

        num_points = torch.full(
            (points.shape[0],), self.num_points, dtype=torch.long, device=points.device
        )

        return points, num_points


class RandomRotatePC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz = xyz_view(points)
        rot_mat = random_rotations(xyz.shape[0], dtype=xyz.dtype, device=xyz.device)
        # Is not performed in place
        xyz @= rot_mat.mT  # (B, N, 3) @ (B, 3, 3) -> (B, N, 3)

        points[..., :3] = xyz
        return points


class RandomRotateAxisPC(nn.Module):
    def __init__(self, axis: str | Sequence[float] = "Z"):
        super().__init__()

        if isinstance(axis, str):
            axis_map = {"X": 0, "Y": 1, "Z": 2}
            if axis not in axis_map:
                raise ValueError(f"Axis must be one of 'X', 'Y', 'Z', got {axis}.")
            rot_vec = torch.zeros((1, 3))
            rot_vec[0, axis_map[axis]] = 1
        else:
            if len(axis) != 3:
                raise ValueError(f"Axis must be a sequence of length 3, got {axis}.")
            rot_vec = torch.tensor([axis], dtype=torch.float32)
            rot_vec /= torch.linalg.vector_norm(rot_vec)

        self.register_buffer("rot_vec", rot_vec)
        self.rot_vec: torch.Tensor  # (1, 3)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        angle = (
            2 * torch.pi * torch.rand((points.shape[0], 1), device=points.device)
        )  # (B, 1)
        rot_mat = axis_angle_to_matrix(angle * self.rot_vec)  # (B, 3) -> (B, 3, 3)

        xyz = xyz_view(points)
        xyz @= rot_mat.mT  # (B, N, 3) @ (B, 3, 3) -> (B, N, 3)

        points[..., :3] = xyz
        return points


class RandomScalePC(ABC, nn.Module):
    def __init__(self, scale_range: tuple[float, float] = (0.8, 1.2)):
        super().__init__()
        self.scale_range = scale_range

    @abstractmethod
    def sample_scale(self, points: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        scale = self.sample_scale(points)

        xyz = xyz_view(points)
        xyz *= scale

        points[..., :3] = xyz
        return points


class RandomUniformScalePC(RandomScalePC):
    def sample_scale(self, points: torch.Tensor) -> torch.Tensor:
        scale = torch.rand((points.shape[0], 1, 1), device=points.device)
        scale = (
            self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * scale
        )
        return scale


class RandomAnisotropicScalePC(RandomScalePC):
    def sample_scale(self, points: torch.Tensor) -> torch.Tensor:
        scale = torch.rand((points.shape[0], 1, 3), device=points.device)
        scale = (
            self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * scale
        )
        return scale


class RandomTranslatePC(nn.Module):
    def __init__(self, max_translate: float = 0.2):
        super().__init__()
        self.max_translate = max_translate

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        translate = torch.rand((points.shape[0], 1, 3), device=points.device)
        translate = 2 * self.max_translate * translate - self.max_translate

        xyz = xyz_view(points)
        xyz += translate

        points[..., :3] = xyz
        return points


class RandomFlipPC(nn.Module):
    def __init__(self, axis: tuple[bool, bool, bool] = (True, True, False)):
        super().__init__()
        tensor_axis = torch.tensor(axis, dtype=torch.float32)

        self.register_buffer("axis", tensor_axis)
        self.axis: torch.Tensor

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        flip = torch.randint(0, 2, (points.shape[0], 1, 3), device=points.device)
        scale = 1 - 2 * flip * self.axis
        xyz = xyz_view(points)
        xyz *= scale

        points[..., :3] = xyz
        return points


SubsamplingTransform = FarthestPointSubSamplePC | nn.Identity
AugmentationTransform = OneOrSequence_T[
    RandomRotatePC
    | RandomRotateAxisPC
    | RandomUniformScalePC
    | RandomAnisotropicScalePC
    | RandomTranslatePC
]
NormalizationTransform = NormalizePC | NormalizeUnitSpherePC
