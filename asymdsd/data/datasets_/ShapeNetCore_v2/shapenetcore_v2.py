import zipfile
from collections.abc import Iterable, Iterator
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from asymdsd.components.common_types import PathLike
from asymdsd.data import (
    ClassLabels,
    DataField,
    DatasetBuilder,
    FieldType,
    PCFieldKey,
)
from asymdsd.data.transforms import DecodeMesh, SampleSurfacePoints

from .synset_map import SYNSET_MAP_55


@lru_cache(1)
def open_zipfile_buffer(data_path):
    return zipfile.ZipFile(data_path, "r")


class ShapeNetCoreV2Builder(DatasetBuilder):
    DATA_FILE = "ShapeNetCore.v2.zip"
    SPLITS = ["train"]
    FILE_FORMAT = "obj"

    SYNSET_MAP_55 = SYNSET_MAP_55

    def __init__(
        self,
        data_path: PathLike | None = None,
        num_pre_sample_points: int = 16384,
        seed: int | None = None,
    ):
        if data_path is None:
            data_path = Path(__file__).parent / "data" / ShapeNetCoreV2Builder.DATA_FILE

        label_names = list(ShapeNetCoreV2Builder.SYNSET_MAP_55.values())

        self._set_info(
            name="ShapeNetCore-v2",
            data_path=data_path,
            splits=ShapeNetCoreV2Builder.SPLITS,
            data_fields=[
                DataField(
                    key=PCFieldKey.POINTS,
                    key_type=FieldType.ARRAY,
                ),
                DataField(
                    key=PCFieldKey.CLOUD_LABEL,
                    key_type=FieldType.STRING_LABEL,
                ),
            ],
            class_labels={PCFieldKey.CLOUD_LABEL: ClassLabels(label_names)},
        )

        self.num_pre_sample_points = num_pre_sample_points
        self.decode_mesh = DecodeMesh(format=ShapeNetCoreV2Builder.FILE_FORMAT)
        self.sample_surface = SampleSurfacePoints(
            num_points=num_pre_sample_points,
            seed=seed,
        )

    def process_instance(self, path: str) -> dict[str, str | np.ndarray]:
        zip_file = open_zipfile_buffer(self.data_path)
        mesh_binary = zip_file.read(path)  # type: ignore
        mesh = self.decode_mesh(mesh_binary)
        points = self.sample_surface(mesh)

        # Rotate axes system such that z is up.
        points[:, :] = points[:, [2, 0, 1]]

        path_parts = path.split("/")
        synset_id = path_parts[-4]

        name = f"{synset_id}_{path_parts[-3].removesuffix(f'.{ShapeNetCoreV2Builder.FILE_FORMAT}')}"
        label = ShapeNetCoreV2Builder.SYNSET_MAP_55[synset_id]

        return {
            "name": name,
            PCFieldKey.POINTS: points,
            PCFieldKey.CLOUD_LABEL: label,
        }

    def iterate_data(
        self, split: str, num_workers: int = 1
    ) -> Iterable[dict[str, str | np.ndarray]]:
        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}")

        return _DataIterator(
            self,
            num_workers=num_workers,
        )


class _DataIterator:
    # TODO: Consider implementing ParallelDataIterator class, implementing init and process_instance
    def __init__(
        self,
        builder: ShapeNetCoreV2Builder,
        num_workers: int | None = 1,
    ):
        self.builder = builder
        self.num_workers = num_workers or 0

        self.paths = [
            path
            for path in zipfile.ZipFile(builder.data_path, "r").namelist()
            if path.endswith(builder.FILE_FORMAT)
        ]
        self.results_len = len(self.paths)

    def __iter__(self) -> Iterator[dict[str, str | np.ndarray]]:
        if self.num_workers > 0:
            with Pool(self.num_workers) as pool:
                self.results = pool.imap_unordered(
                    self.builder.process_instance, self.paths, chunksize=4
                )
                for result in self.results:
                    yield result
        else:
            for path in self.paths:
                yield self.builder.process_instance(path)

    def __len__(self):
        return self.results_len
