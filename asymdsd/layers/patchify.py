from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import torch
from pytorch3d.ops import (
    ball_query,
    knn_gather,
    knn_points,
    sample_farthest_points,
)
from torch import nn

from ..components.common_types import OptionalListTensor, OptionalTensor
from ..components.utils import xyz_view


@dataclass
class PatchPoints:
    points: torch.Tensor
    num_points: OptionalTensor = None
    patches_idx: OptionalListTensor = None
    centers_idx: OptionalListTensor = None


class MultiPatches(NamedTuple):
    patches: torch.Tensor
    patches_idx: list[torch.Tensor]
    centers: list[torch.Tensor]


class PointPatchify(nn.Module):
    def __init__(
        self,
        num_patches: int = 64,
        patch_size: int = 32,
        limit_radius: float | None = None,
    ):
        super().__init__()
        self.num_patches = num_patches  # P
        self.patch_size = patch_size  # K
        self.limit_radius = limit_radius

        self.sample_patch = (  # type: ignore
            partial(knn_points, return_sorted=True)
            if limit_radius is None
            else partial(ball_query, radius=limit_radius)
        )

    def sample_patch(
        self,
        centers: torch.Tensor,
        points: torch.Tensor,
        K: int,
        return_nn: bool,
    ): ...

    def forward(
        self, points: torch.Tensor, num_points: OptionalTensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = xyz_view(points)

        # Can set random_start_point to True, but assume permuted points
        centers, idx = sample_farthest_points(
            xyz,
            lengths=num_points,
            K=self.num_patches,
            random_start_point=True,
        )  # (B, P, 3)
        knn_res = self.sample_patch(centers, xyz, K=self.patch_size, return_nn=False)

        idx: torch.Tensor = knn_res.idx  # type: ignore
        # (B, P, K)

        # Can use KNN_GATHER if needed
        idx = idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1])
        # (B, P, K, >3)
        points = points.unsqueeze(2).expand(-1, -1, self.patch_size, -1)
        # (B, N, K, >3)

        patches = torch.gather(points, dim=1, index=idx)  # (B, P, K, >3)

        xyz_patches = xyz_view(patches)
        xyz_patches -= centers.unsqueeze(-2)  # Center points normalization

        # Center point is included in patch. This means that after normalization,
        # each patch has a point that is [0,0,0,...]
        return patches, centers


class PointPatchifyIdx(nn.Module):
    def __init__(
        self,
        num_patches: int = 64,
        patch_size: int = 32,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size

    def forward(
        self, points: torch.Tensor, num_points: OptionalTensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Can set random_start_point to True, but assume permuted points
        centers, _ = sample_farthest_points(
            points,
            lengths=num_points,
            K=self.num_patches,
            random_start_point=True,
        )  # (B, P, 3)
        knn_res = knn_points(
            centers,
            points,
            lengths2=num_points,
            K=self.patch_size,
            return_nn=False,
            return_sorted=True,
        )

        idx: torch.Tensor = knn_res.idx  # (B, P, K)

        return idx, centers


class MultiPointPatchify(nn.Module):
    def __init__(
        self,
        num_patches: list[int] = [2048, 64],
        patch_size: list[int] = [64, 32],
    ):
        super().__init__()

        num_patches = list(num_patches)
        patch_size = list(patch_size)

        if len(num_patches) != len(patch_size) and len(num_patches) < 1:
            raise ValueError(
                "Length of num_patches and patch_size must be the same and greater than 0"
            )

        self.patchify = nn.ModuleList(
            [
                PointPatchifyIdx(num_patches[i], patch_size[i])
                for i in range(len(num_patches))
            ]
        )

        self.center_patches = CenterPatches()

    def forward(self, patch_points: PatchPoints) -> MultiPatches:
        points = patch_points.points
        num_points = patch_points.num_points

        patches_idx = []
        patch_centers = []

        # First patches are immediately gathered, as there no embedding involved yet.
        idx, centers = self.patchify[0](points, num_points)
        patches = knn_gather(points, idx, lengths=num_points)
        patches = self.center_patches(patches, centers)

        patch_centers.append(centers)

        for i in range(1, len(self.patchify)):
            idx, centers = self.patchify[i](centers)
            patches_idx.append(idx)
            patch_centers.append(centers)

        return MultiPatches(patches, patches_idx, patch_centers)


class ToMultiPatches(nn.Module):
    # NOTE: This module does not support variable length patches
    def __init__(self):
        super().__init__()
        self.center_patches = CenterPatches()

    def forward(
        self,
        patch_points: PatchPoints,
    ) -> MultiPatches:
        points = patch_points.points
        patches_idx: list[torch.Tensor] = patch_points.patches_idx  # type: ignore
        centers_idx: list[torch.Tensor] = patch_points.centers_idx  # type: ignore

        patch_centers = []

        patches = knn_gather(points, patches_idx[0])
        centers = points.gather(1, centers_idx[0].unsqueeze(-1).expand(-1, -1, 3))

        patches = self.center_patches(patches, centers)

        patch_centers.append(centers)

        for i in range(1, len(patches_idx)):
            centers = points.gather(1, centers_idx[i].unsqueeze(-1).expand(-1, -1, 3))
            patch_centers.append(centers)

        return MultiPatches(patches, patches_idx, patch_centers)


class CenterPatches(nn.Module):
    def forward(self, patches: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        xyz_patches = xyz_view(patches)
        xyz_patches -= centers.unsqueeze(-2)  # Center points normalization

        return patches
