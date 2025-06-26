import fpsample as fps
import numpy as np
from scipy.spatial import KDTree

from .transforms import RandomizedTransform


class PatchifyPC(RandomizedTransform):
    def __init__(
        self,
        num_patches: int = 64,
        patch_size: int = 32,
        deterministic: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed, batched=False)
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.deterministic = deterministic

    def _get_start_idx(self, num_points: int) -> int:
        if self.deterministic:
            return num_points // 2
        return self.generator.integers(num_points)

    def transform(self, points: np.ndarray) -> dict[str, list[np.ndarray]]:
        start_idx = self._get_start_idx(points.shape[0])
        # TODO: Potentially Add suport for features, or remove patches
        center_idx = fps.bucket_fps_kdline_sampling(
            points[..., :3],
            self.num_patches,
            h=7,
            start_idx=start_idx,
        )
        centers = points[center_idx]

        kdtree = KDTree(points[..., :3])

        dist, idx = kdtree.query(centers, self.patch_size)

        return {
            # TODO: Check centering
            # "patches": points[idx] - centers[:, None],
            "centers_idx": [center_idx.astype(np.int64)],
            "patches_idx": [idx.astype(np.int64)],
        }


PatchifyModule = PatchifyPC  # Typing alias for PatchifyPC (could add more later)
