from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from ..components.common_types import OneOrSequence_T


class PCTransform(ABC):
    def __init__(self, batched: bool = False) -> None:
        self.set_batched(batched)

    def set_batched(self, batched: bool) -> None:
        self.batched = batched
        if batched:
            self._call = self.transform
        else:
            self._call = self.batchify_transform

    def batchify_transform(self, s_points: np.ndarray) -> np.ndarray:
        return self.transform(np.array([s_points]))[0]

    def __call__(self, examples: np.ndarray) -> np.ndarray:
        return self._call(examples)

    @abstractmethod
    def transform(self, points: np.ndarray) -> np.ndarray:
        # This should be a transform that applies to a batch of point clouds
        # (B, N, C >= 3)
        pass


class RandomizedPCTransform(PCTransform):
    def __init__(self, seed: int | None = None, batched: bool = False) -> None:
        super().__init__(batched=batched)
        self.set_seed(seed)

    def set_seed(self, seed: int | None = None):
        self.seed = seed
        self.generator = np.random.default_rng(seed)


class SelectFeaturesPC(PCTransform):
    def __init__(self, features: Sequence[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.features = features

    def transform(self, points: np.ndarray) -> np.ndarray:
        return points[..., self.features]


class CenterPC(PCTransform):
    def transform(self, points: np.ndarray) -> np.ndarray:
        center = points[..., :3].mean(axis=-2, keepdims=True)
        points[..., :3] -= center

        return points


class NormalizeUnitSpherePC(PCTransform):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.center_pc = CenterPC(batched=True)

    def transform(self, points: np.ndarray) -> np.ndarray:
        points = self.center_pc(points)

        norm = np.linalg.norm(points[..., :3], axis=-1, keepdims=True)
        scale = norm.max(axis=(-1, -2))

        points[..., :3] /= scale

        return points

class NormalizePC(PCTransform):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.center_pc = CenterPC()

    def transform(self, points: np.ndarray) -> np.ndarray:
        points = self.center_pc(points)

        std = points[..., :3].std(axis=(-1, -2), keepdims=True)
        points[..., :3] /= std

        return points


class RandomRotatePC(RandomizedPCTransform):
    def transform(self, points: np.ndarray) -> np.ndarray:
        rot_mat = Rotation.random(points.shape[0], random_state=self.seed).as_matrix()
        points[..., :3] @= rot_mat.transpose(0, 2, 1)

        return points


class RandomRotateAxisPC(RandomizedPCTransform):
    def __init__(self, axis: str | Sequence[float] = "Z", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(axis, str):
            axis_map = {"X": 0, "Y": 1, "Z": 2}
            if axis not in axis_map:
                raise ValueError(f"Axis must be one of 'X', 'Y', 'Z', got {axis}.")
            self.rot_vec = np.zeros((1, 3))
            self.rot_vec[0, axis_map[axis]] = 1
        else:
            if len(axis) != 3:
                raise ValueError(f"Axis must be a sequence of length 3, got {axis}.")
            self.rot_vec = np.array(axis)
            self.rot_vec /= np.linalg.norm(self.rot_vec)

    def transform(self, points):
        angle = self.generator.uniform(0, 2 * np.pi, (points.shape[0], 1))
        rot_mat = Rotation.from_rotvec(angle * self.rot_vec).as_matrix()

        points[..., :3] @= rot_mat.transpose(0, 2, 1)

        return points


class RandomUniformScalePC(RandomizedPCTransform):
    def __init__(
        self, scale_range: tuple[float, float] = (0.8, 1.2), *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.scale_range = scale_range

    def transform(self, points: np.ndarray) -> np.ndarray:
        scale = self.generator.uniform(
            self.scale_range[0], self.scale_range[1], (points.shape[0], 1, 1)
        )
        points[..., :3] *= scale

        return points


class RandomAnisotropicScalePC(RandomizedPCTransform):
    def __init__(
        self, scale_range: tuple[float, float] = (0.8, 1.2), *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.scale_range = scale_range

    def transform(self, points: np.ndarray) -> np.ndarray:
        scale = self.generator.uniform(
            self.scale_range[0], self.scale_range[1], (points.shape[0], 1, 3)
        )
        points[..., :3] *= scale

        return points


class RandomFlipPC(RandomizedPCTransform):
    def __init__(
        self, axis: tuple[bool, bool, bool] = (True, True, False), *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.axis = np.array(axis)

    def transform(self, points: np.ndarray) -> np.ndarray:
        flip = self.generator.choice([0, 1], (points.shape[0], 1, 3))
        scale = 1 - 2 * flip * self.axis
        points[..., :3] *= scale

        return points


class RandomTranslatePC(RandomizedPCTransform):
    def __init__(self, max_translate: float = 0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_translate = max_translate

    def transform(self, points: np.ndarray) -> np.ndarray:
        # TODO: Only works for batched input
        translate = self.generator.uniform(
            -self.max_translate, self.max_translate, (points.shape[0], 1, 3)
        )

        points[..., :3] += translate

        return points


class UniformSubSamplePC(RandomizedPCTransform):
    def __init__(self, num_points: int = 1024, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_points = num_points

    def transform(self, points: np.ndarray) -> np.ndarray:
        new_points = self.generator.choice(
            points, self.num_points, axis=-2, replace=False
        )
        return new_points


NormalizationTransform = NormalizePC | NormalizeUnitSpherePC
AugmentationTransform = OneOrSequence_T[RandomizedPCTransform]