from abc import ABC, abstractmethod

import torch
from pytorch3d.ops import knn_points
from torch import nn


class PatchMaskingGenerator(nn.Module, ABC):
    def __init__(
        self,
        mask_ratio: float | tuple[float, float] = 0.5,
        multi_mask: int | None = None,
        multi_block: int | None = None,
    ) -> None:
        super().__init__()
        self._mask_ratio = (
            mask_ratio if isinstance(mask_ratio, tuple) else (mask_ratio, mask_ratio)
        )
        self._multi_mask = 1 if multi_mask is None else multi_mask
        self._multi_block = multi_block

    @property
    def mask_ratio(self) -> tuple[float, float]:
        return self._mask_ratio

    @property
    def multi_mask(self) -> int:
        return self._multi_mask

    @property
    def multi_block(self) -> int | None:
        return self._multi_block

    def sample_ratio(self, ratio: tuple[float, float]) -> float:
        return torch.rand(1).item() * (ratio[1] - ratio[0]) + ratio[0]

    def sample_mask_ratio(self) -> float:
        return self.sample_ratio(self.mask_ratio)

    @abstractmethod
    def forward(
        self, centers: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        pass


class RandomPatchMasking(PatchMaskingGenerator):
    # TODO: Implement multi_block
    def forward(self, centers: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        B, P = centers.shape[:2]

        mask_ratio = self.sample_mask_ratio()
        num_masks = round(mask_ratio * P)

        if num_masks == 0:
            return None, None

        mask_shape = (B * self.multi_mask, P)

        # torch.randperm does not support sized output, this is a tensor alternative:
        rand_uniform = torch.rand(mask_shape, device=centers.device)
        mask_indices = rand_uniform.argsort(dim=-1)[..., :num_masks]

        mask = torch.zeros(mask_shape, dtype=torch.bool, device=centers.device)
        mask.scatter_(-1, mask_indices, True)

        return mask, None


class BlockPatchMasking(PatchMaskingGenerator):
    def __init__(
        self,
        mask_ratio: float | tuple[float, float] = 0.5,
        multi_mask: int | None = None,
        multi_block: int | None = None,
        block_ratio: float | tuple[float, float] = 0.2,
        adjust_ratio: float = 0.1,
        inverse_block_masking: bool = False,
    ) -> None:
        super().__init__(
            mask_ratio=mask_ratio, multi_mask=multi_mask, multi_block=multi_block
        )
        self.block_ratio = (
            block_ratio
            if isinstance(block_ratio, tuple)
            else (block_ratio, block_ratio)
        )
        self.adjust_ratio = adjust_ratio
        self.inverse_block_masking = inverse_block_masking

        if inverse_block_masking and multi_block is not None:
            raise ValueError(
                "Inverse block masking should not be used with multi_block"
            )

        # if (
        #     self.multi_block is not None
        #     and block_ratio * self.multi_block > self.mask_ratio[0] + 1e-6
        # ):
        #     raise ValueError(
        #         f"block_ratio={block_ratio} * multi_block={multi_block} "
        #         f"= {block_ratio * multi_block} "  # type: ignore
        #         f"should not exceed mask_ratio={self.mask_ratio[0]}"
        #     )

        # Only needs to flip mask if using standard block masking
        self.process_mask = nn.Identity() if inverse_block_masking else lambda x: ~x

    def forward(
        self, centers: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        B, P, F = centers.shape  # C should be 3

        mask_ratio = self.sample_mask_ratio()
        num_masks = round(mask_ratio * P)

        if num_masks == 0:
            return None, None

        block_ratio = self.sample_ratio(self.block_ratio)
        block_size = int(block_ratio * P)

        ratio = 1 - mask_ratio if self.inverse_block_masking else mask_ratio
        adjust_ratio = (
            self.adjust_ratio if self.inverse_block_masking else -self.adjust_ratio
        )

        block_fraction = (ratio + adjust_ratio) / block_size

        num_centers = round(P * block_fraction)

        mask_shape = (B * self.multi_mask, P)

        rand_uniform = torch.rand(mask_shape, device=centers.device)
        center_indices = rand_uniform.argsort(dim=-1)[..., :num_centers]

        center_indices = center_indices.unsqueeze(-1).expand(-1, -1, F)
        # Repeat interleave to match flatten order after expanding (unlike tile)
        centers = centers.repeat_interleave(self.multi_mask, dim=0)

        selected_centers = torch.gather(
            centers, 1, center_indices
        )  # (B*multi_mask, num_centers, 3)

        knn_res = knn_points(
            selected_centers, centers, K=block_size, return_sorted=False
        )

        # (B*multi_mask, num_centers*block_size)
        idx: torch.Tensor = torch.flatten(knn_res.idx, -2, -1)

        mask = torch.ones(mask_shape, dtype=torch.bool, device=centers.device)
        mask.scatter_(-1, idx, False)  # These patches are not masked

        mask = self.process_mask(mask)  # Inverse mask if needed

        # --- Now ensure each input has exactly num_masks masks
        rand_uniform = torch.rand(mask_shape, device=centers.device)
        # Randomly order masks for selection, but sorted from True to False
        mask_flip_rand = torch.where(mask, -rand_uniform, rand_uniform)
        mask_indices = mask_flip_rand.argsort(dim=-1)[..., :num_masks]

        final_mask = torch.zeros(mask_shape, dtype=torch.bool, device=centers.device)
        final_mask.scatter_(-1, mask_indices, True)

        block_indices = knn_res.idx[:, : self.multi_block] if self.multi_block else None

        return final_mask, block_indices


class InverseBlockPatchMasking(BlockPatchMasking):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, inverse_block_masking=True)


MaskGenerator = RandomPatchMasking | BlockPatchMasking | InverseBlockPatchMasking

if __name__ == "__main__":
    rand_mask_generator = RandomPatchMasking(0.6, 8)
    inv_block_mask_generator = InverseBlockPatchMasking(0.4)
    x = torch.randn(32, 64, 3)
    mask = rand_mask_generator(x)
    mask = inv_block_mask_generator(x)
