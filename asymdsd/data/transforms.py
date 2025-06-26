from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Callable, Sequence

import fpsample as fps
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


class Transform(ABC):
    def __init__(self, batched: bool = False) -> None:
        self.batched = batched
        if not batched:
            self._call = self.transform
        else:
            self._call = self.list_transform

    def list_transform(self, examples: list[Any]) -> list[Any]:
        return [self.transform(*ex) for ex in zip(examples)]

    def __call__(self, *args, **kwargs) -> Any:
        return self._call(*args, **kwargs)

    @abstractmethod
    def transform(self, examples: Any) -> Any:
        pass


class RandomizedTransform(Transform):
    def __init__(self, seed: int | None = None, batched: bool = False) -> None:
        super().__init__(batched=batched)
        self.set_seed(seed)

    def set_seed(self, seed: int | None = None):
        self.seed = seed
        self.generator = np.random.default_rng(seed)


# TODO: Make a class specific for mapping array_dicts.


class MapColumn:
    def __init__(
        self,
        transform: Callable | Transform | Sequence[Callable | Transform],
        input_columns: str | list[str],
        output_columns: str | list[str] | None = None,
        remove_columns: str | list[str] | None = None,
        input_as_positional_args: bool = True,
    ) -> None:
        """A class to map a transform to a column in a dataset.

        Args:
            transform (Callable | Transform | Sequence[Callable | Transform]): The transform to apply to the input column.
            input_column (str | list[str]): The column name(s) to select to provide as positional arguments to the transform.
            output_column (str | list[str] | None, optional): The column name(s) to store the output of the transform. If None, the output will be stored in the input column. Defaults to None.
        """
        self.input_columns = (
            [input_columns] if isinstance(input_columns, str) else input_columns
        )
        self.output_column = (
            [output_columns] if isinstance(output_columns, str) else output_columns
        )
        self.remove_columns = (
            [remove_columns] if isinstance(remove_columns, str) else remove_columns
        )

        self.input_as_positional_args = input_as_positional_args

        self.transform: Callable | Transform = transform  # type: ignore
        if isinstance(self.transform, Sequence):
            self.transform = Compose(self.transform)

    def __call__(self, examples_dict: dict[str, Any]) -> dict[str, Any]:
        if self.input_as_positional_args:
            output = self.transform(*[examples_dict[col] for col in self.input_columns])
        else:
            input_dict = {col: examples_dict[col] for col in self.input_columns}
            output = self.transform(input_dict)

        if self.remove_columns is not None:
            for col in self.remove_columns:
                del examples_dict[col]

        if self.output_column is None:
            examples_dict.update(output)
        else:
            output = [output] if not isinstance(output, tuple) else output
            for col, out in zip(self.output_column, output):
                examples_dict[col] = out

        return examples_dict


class Compose:
    def __init__(self, transforms: Sequence[Callable | Transform]) -> None:
        self.transforms = transforms

    def __call__(self, examples: Any) -> Any:
        for transform in self.transforms:
            examples = transform(examples)
        return examples


class DecodeMesh(Transform):
    def __init__(self, format: str, batched: bool = False) -> None:
        super().__init__(batched=batched)
        self.format = format

    def transform(self, mesh_binary: bytes) -> trimesh.Trimesh:
        stream = BytesIO(mesh_binary)
        mesh: trimesh.Trimesh | trimesh.Scene = trimesh.load_mesh(stream, self.format)  # type: ignore
        # Convert scene of trimeshes to single trimesh
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)  # type: ignore
        return mesh  # type: ignore


class EncodeArray(Transform):
    def transform(self, array: np.ndarray) -> bytes:
        stream = BytesIO()
        np.save(stream, array)
        return stream.getvalue()


class DecodeArray(Transform):
    def transform(self, array_binary: bytes) -> np.ndarray:
        stream = BytesIO(array_binary)
        return np.load(stream)


class BinaryArrayProcessor(Transform):
    def __init__(
        self,
        transfrom: Callable[[np.ndarray], np.ndarray],
        batched: bool = False,
    ) -> None:
        super().__init__(batched=batched)
        self.array_transform = transfrom
        self.encoder = EncodeArray(batched=False)
        self.decoder = DecodeArray(batched=False)

    def transform(self, array_binary: bytes) -> bytes:
        x = self.decoder(array_binary)
        x = self.array_transform(x)
        x = self.encoder(x)
        return x


class ToNumpyBatch(Transform):
    def __init__(self):
        super().__init__(batched=False)

    def transform(self, array: list[np.ndarray]) -> np.ndarray:
        return np.array(array)


class ToListBatch(Transform):
    def __init__(self):
        super().__init__(batched=False)

    def transform(self, array: np.ndarray) -> list[np.ndarray]:
        return list(array)


class SampleSurfacePoints(Transform):
    def __init__(
        self,
        num_points: int = 1024,
        dtype=np.float32,
        batched: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(batched=batched)
        self.num_points = num_points
        self.dtype = dtype
        self.seed = seed

    def transform(self, mesh: trimesh.Trimesh) -> np.ndarray:
        # Returns TrackedArray
        points, _ = trimesh.sample.sample_surface(  # type: ignore
            mesh, self.num_points, seed=self.seed
        )
        points = np.array(points, dtype=self.dtype)
        return points


class UniformSampleArrays(RandomizedTransform):
    def __init__(
        self,
        sample_size: int,
        axis: int = 0,
        deterministic: bool = False,
        batched: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(batched=batched, seed=seed)
        self.num_samples = sample_size
        self.axis = axis
        self.deterministic = deterministic
        self.generator = np.random.default_rng(seed=self.seed)

    def transform(self, arrays_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        arrays_shape = arrays_dict[list(arrays_dict.keys())[0]].shape
        if arrays_shape[self.axis] <= self.num_samples:
            return arrays_dict
        indices_shape = arrays_shape[: self.axis] + (self.num_samples,)
        indices = self.generator.choice(
            arrays_shape[self.axis],
            indices_shape,
            replace=False,
        )
        if self.deterministic:
            self.generator = np.random.default_rng(seed=self.seed)
        return {k: np.take(v, indices, axis=self.axis) for k, v in arrays_dict.items()}


class FarthestPointSampleArrays(RandomizedTransform):
    def __init__(
        self,
        sample_size: int,
        axis: int = -2,
        points_key: str = "points",
        deterministic: bool = False,
        batched: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(batched=batched, seed=seed)
        self.num_samples = sample_size
        self.axis = axis
        self.points_key = points_key
        self.deterministic = deterministic

    def _get_start_idx(self, num_points: int) -> int:
        if self.deterministic:
            return num_points // 2
        return self.generator.integers(num_points)

    def transform(self, arrays_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        arrays_shape = arrays_dict[list(arrays_dict.keys())[0]].shape
        num_points = arrays_shape[self.axis]
        if num_points <= self.num_samples:
            return arrays_dict
        start_idx = self._get_start_idx(num_points)
        indices = fps.bucket_fps_kdline_sampling(
            arrays_dict[self.points_key], self.num_samples, h=7, start_idx=start_idx
        ).astype(np.int32)
        return {k: np.take(v, indices, axis=self.axis) for k, v in arrays_dict.items()}


class CropSampleArrays(RandomizedTransform):
    def __init__(
        self,
        num_points_range: tuple[int | None, int | None] = (1024, 1024),
        crop_scale: tuple[float, float] = (0.4, 1.0),  # type: ignore
        aspect_ratio: tuple[float, float] = (0.33, 3.0),
        points_key: str = "points",
        batched: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(batched=batched, seed=seed)
        self.min_num_points = num_points_range[0]
        self.max_num_points = num_points_range[1]
        self.crop_scale = crop_scale
        self.aspect_ratio = aspect_ratio
        self.points_key = points_key

        if crop_scale[0] > crop_scale[1]:
            raise ValueError(f"crop_scale must be (min, max), got {crop_scale}.")
        if crop_scale[0] < 0 or crop_scale[1] > 1:
            raise ValueError(f"crop_scale must be in [0, 1], got {crop_scale}.")

        if aspect_ratio[0] > aspect_ratio[1]:
            raise ValueError(f"aspect_ratio must be (min, max), got {aspect_ratio}.")
        if aspect_ratio[0] < 0:
            raise ValueError(f"aspect_ratio must be positive, got {aspect_ratio}.")

    def transform(self, arrays_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        points = arrays_dict[self.points_key]
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

        crop = {k: v[gather_indices] for k, v in arrays_dict.items()}

        return crop


class UniformVoxelSampleArrays(RandomizedTransform):
    def __init__(
        self,
        voxel_size: int,
        axis: int = -2,
        batched: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(batched=batched, seed=seed)
        self.voxel_size = voxel_size
        self.axis = axis

    def transform(self, arrays_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError("UniformVoxelSubSampleArrays not implemented")


class PadArrays(Transform):
    def __init__(
        self,
        pad_to_length: int,
        axis: int = 0,
        batched: bool = False,
        output_arr_len_key: str = "arr_len",
    ) -> None:
        super().__init__(batched=batched)
        self.pad_to_length = pad_to_length
        self.axis = axis
        self.output_arr_len_key = output_arr_len_key

    def transform(
        self, arrays_dict: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray | int]:
        arr_len = arrays_dict[list(arrays_dict.keys())[0]].shape[self.axis]
        padded_arrays_dict: dict[str, np.ndarray | int] = {
            k: self._pad_array(v) for k, v in arrays_dict.items()
        }
        padded_arrays_dict[self.output_arr_len_key] = arr_len
        return padded_arrays_dict

    def _pad_array(self, array: np.ndarray) -> np.ndarray:
        assert (
            array.shape[self.axis] <= self.pad_to_length
        ), "pad_to_length must be greater than or equal to the length of the axis"
        pad_width = [
            (0, self.pad_to_length - array.shape[self.axis])
            if dim == self.axis or dim == self.axis + array.ndim
            else (0, 0)
            for dim in range(array.ndim)
        ]
        return np.pad(array, pad_width)


class PadOrSubSampleArrays(RandomizedTransform):
    def __init__(
        self,
        max_array_size: int,
        axis: int = 0,
        output_arr_len_key: str = "arr_len",
        batched: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(batched=batched, seed=seed)
        self.max_array_size = max_array_size
        self.axis = axis
        self.output_arr_len_key = output_arr_len_key

        self.subsample = UniformSampleArrays(
            sample_size=self.max_array_size,
            axis=self.axis,
            seed=self.seed,
        )

        self.pad = PadArrays(
            pad_to_length=self.max_array_size,
            axis=self.axis,
        )

    def transform(
        self, arrays_dict: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray | int]:
        arr_len = arrays_dict[list(arrays_dict.keys())[0]].shape[self.axis]

        if arr_len > self.max_array_size:
            arrays_dict[self.output_arr_len_key] = self.max_array_size  # type: ignore
            arrays_dict = self.subsample(arrays_dict)

        elif arr_len < self.max_array_size:
            arrays_dict = self.pad(arrays_dict)

        return arrays_dict  # type: ignore
