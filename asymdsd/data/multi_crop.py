from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from jsonargparse import lazy_instance
from scipy.spatial.transform import Rotation

from .dataset_utils import compose_transform
from .pc_transforms import PCTransform, RandomRotateAxisPC

# TODO: Move
_DEFAULT_AUGMENTATION_TRANSFORM = lazy_instance(RandomRotateAxisPC)


class SampleCropPC:
    def __init__(
        self,
        num_points_range: tuple[int | None, int | None] = (1024, 1024),
        crop_scale: tuple[float, float] = (0.4, 1.0),  # type: ignore
        aspect_ratio: tuple[float, float] = (0.33, 3.0),
        seed: int | None = None,
    ) -> None:
        self.min_num_points = num_points_range[0]
        self.max_num_points = num_points_range[1]
        self.crop_scale = crop_scale
        self.aspect_ratio = aspect_ratio

        if crop_scale[0] > crop_scale[1]:
            raise ValueError(f"crop_scale must be (min, max), got {crop_scale}.")
        if crop_scale[0] < 0 or crop_scale[1] > 1:
            raise ValueError(f"crop_scale must be in [0, 1], got {crop_scale}.")

        if aspect_ratio[0] > aspect_ratio[1]:
            raise ValueError(f"aspect_ratio must be (min, max), got {aspect_ratio}.")
        if aspect_ratio[0] < 0:
            raise ValueError(f"aspect_ratio must be positive, got {aspect_ratio}.")

        self.seed = seed
        self.generator = np.random.default_rng(seed)

    def __call__(
        self, points: np.ndarray, features_dict: dict[str, np.ndarray] | None = None
    ) -> dict[str, np.ndarray]:
        P, F = points.shape

        rot_mat = Rotation.random(random_state=self.generator).as_matrix()
        rot_points = points @ rot_mat

        scale = self.generator.uniform(self.crop_scale[0], self.crop_scale[1])
        gather_num_points = int(scale * P)
        gather_num_points = np.clip(gather_num_points, self.min_num_points, None)

        aspect_ratio = self.generator.uniform(
            self.aspect_ratio[0], self.aspect_ratio[1]
        )
        scales = np.array([1, 1 / np.sqrt(aspect_ratio), np.sqrt(aspect_ratio)])
        scales = self.generator.permutation(scales)

        center_point = rot_points[self.generator.integers(P)]

        dist_l1 = np.max(np.abs(scales * (rot_points - center_point)), axis=-1)
        gather_indices = np.argsort(dist_l1)[:gather_num_points]

        if self.max_num_points and gather_num_points > self.max_num_points:
            gather_indices = self.generator.choice(
                gather_indices, self.max_num_points, replace=False
            )

        crop = {"points": points[gather_indices]}
        if features_dict:
            for k, v in features_dict.items():
                crop[k] = v[gather_indices]

        return crop


@dataclass
class CropConfig:
    num_crops: int = 1
    num_points_range: tuple[int | None, int | None] = (1024, 1024)
    scale: float | tuple[float, float] = (0.4, 1.0)  # Set to 1.0 for no crop
    aspect_ratio: tuple[float, float] = (0.33, 3.0)
    pre_crop_transform: PCTransform | Sequence[PCTransform] | None = (
        _DEFAULT_AUGMENTATION_TRANSFORM
    )

    def __post_init__(self) -> None:
        self.scale = (
            self.scale if isinstance(self.scale, tuple) else (self.scale, self.scale)
        )


@dataclass
class MultiCropConfig:
    global_cfg: CropConfig = field(default_factory=lambda: CropConfig(2))
    local_cfg: CropConfig | None = None
    sequential_cfg: CropConfig | None = None


class PointMultiCrop:
    def __init__(
        self,
        multi_crop_config: MultiCropConfig,
        *,
        seed: int | None = None,
    ):
        self.mc_cfg = multi_crop_config
        self.global_cfg = global_cfg = self.mc_cfg.global_cfg
        self.local_cfg = local_cfg = self.mc_cfg.local_cfg
        self.sequential_cfg = sequential_cfg = self.mc_cfg.sequential_cfg

        self.seed = seed
        self.generator = np.random.default_rng(seed)

        # Legacy of old implementation
        def _get_sample_fn(crop_cfg: CropConfig) -> Callable:
            # TODO: if scale is (1.0, 1.0) sample to range
            return SampleCropPC(
                crop_cfg.num_points_range,
                crop_scale=crop_cfg.scale,  # type: ignore
                seed=self.seed,
            )

        self.sample_crop_global = _get_sample_fn(global_cfg)
        self.global_transform = compose_transform(
            global_cfg.pre_crop_transform, seed=self.seed
        )

        if local_cfg:
            self.sample_crop_local = _get_sample_fn(local_cfg)
            self.local_transform = compose_transform(
                local_cfg.pre_crop_transform, seed=self.seed
            )

        if sequential_cfg:
            self.sample_crop_sequential = _get_sample_fn(sequential_cfg)
            self.sequential_transform = compose_transform(
                sequential_cfg.pre_crop_transform, seed=self.seed
            )

    def multi_crop_sample(
        self,
        points: np.ndarray,
        num_crops: int,
        transform: Callable[[np.ndarray], np.ndarray],
        sample_fn: Callable[[np.ndarray, dict | None], np.ndarray],
        features_dict: dict[str, np.ndarray] | None = None,
    ) -> list[dict[str, np.ndarray]]:
        crops = []
        for _ in range(num_crops):
            points = transform(points)
            crop = sample_fn(points, features_dict)
            crops.append(crop)
        return crops

    def sequential_crop_sample(
        self,
        points: np.ndarray,
        num_crops: int,
        transform: Callable[[np.ndarray], np.ndarray],
        sample_fn: Callable[[np.ndarray, dict | None], np.ndarray],
        features_dict: dict[str, np.ndarray] | None = None,
    ) -> list[dict[str, np.ndarray]]:
        points = transform(points)
        return [sample_fn(points, features_dict) for _ in range(num_crops)]

    # Change to arrays dict
    def __call__(
        self, points: np.ndarray, features_dict: dict[str, np.ndarray] | None = None
    ) -> dict[str, list[dict[str, np.ndarray]]]:
        crop_dict = {}
        cfg = self.mc_cfg

        global_crops = self.multi_crop_sample(
            points,
            cfg.global_cfg.num_crops,
            self.global_transform,
            self.sample_crop_global,
            features_dict,
        )
        crop_dict["global_crops"] = global_crops

        if cfg.local_cfg and cfg.local_cfg.num_crops > 0:
            local_crops = self.multi_crop_sample(
                points,
                cfg.local_cfg.num_crops,
                self.local_transform,
                self.sample_crop_local,
                features_dict,
            )
            crop_dict["local_crops"] = local_crops

        if cfg.sequential_cfg and cfg.sequential_cfg.num_crops > 0:
            sequential_crops = self.sequential_crop_sample(
                points,
                cfg.sequential_cfg.num_crops,
                self.sequential_transform,
                self.sample_crop_sequential,
                features_dict,
            )
            crop_dict["sequential_crops"] = sequential_crops

        return crop_dict
