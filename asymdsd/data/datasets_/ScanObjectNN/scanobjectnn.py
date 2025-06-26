import zipfile
from collections.abc import Iterable, Iterator
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

from asymdsd.components.common_types import PathLike
from asymdsd.data import (
    ClassLabels,
    DataField,
    DatasetBuilder,
    FieldType,
    PCFieldKey,
)

from .label_names import LABEL_NAMES


@lru_cache(1)
def open_h5f(zip_path, h5f_path):
    zip_file = zipfile.ZipFile(zip_path, "r")
    file = zip_file.open(h5f_path)
    h5f = h5py.File(file, "r")
    return h5f


class ScanObjectNNBuilder(DatasetBuilder):
    DATA_FILE = "h5_files.zip"
    SPLITS = [
        "OBJ_ONLY_train",
        "OBJ_ONLY_test",
        "OBJ_BG_train",
        "OBJ_BG_test",
        "PB_T50_RS_train",
        "PB_T50_RS_test",
    ]
    H5_FILE_MAP = {
        "OBJ_ONLY_train": "h5_files/main_split_nobg/training_objectdataset.h5",
        "OBJ_ONLY_test": "h5_files/main_split_nobg/test_objectdataset.h5",
        "OBJ_BG_train": "h5_files/main_split/training_objectdataset.h5",
        "OBJ_BG_test": "h5_files/main_split/test_objectdataset.h5",
        "PB_T50_RS_train": "h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5",
        "PB_T50_RS_test": "h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5",
    }

    LABEL_NAMES = LABEL_NAMES

    def __init__(
        self,
        data_path: PathLike | None = None,
        # num_pre_sample_points: int = 16384,
        # seed: int | None = None,
    ):
        if data_path is None:
            data_path = Path(__file__).parent / "data" / ScanObjectNNBuilder.DATA_FILE

        self._set_info(
            name="ScanObjectNN",
            data_path=data_path,
            splits=ScanObjectNNBuilder.SPLITS,
            data_fields=[
                DataField(
                    key=PCFieldKey.POINTS,
                    key_type=FieldType.ARRAY,
                ),
                DataField(
                    key=PCFieldKey.CLOUD_LABEL,
                    key_type=FieldType.INT_LABEL,
                ),
            ],
            class_labels={
                PCFieldKey.CLOUD_LABEL: ClassLabels(ScanObjectNNBuilder.LABEL_NAMES)
            },
        )

    def process_instance(self, args: tuple[str, int]) -> dict[str, str | np.ndarray]:
        h5f_path, index = args
        h5f = open_h5f(self.data_path, h5f_path)
        data = h5f["data"]
        label = h5f["label"]

        points = np.array(data[index])  # type: ignore
        # Rotate axes system such that z is up.
        points[:, :] = points[:, [2, 0, 1]]

        return {
            "name": str(index),
            PCFieldKey.POINTS: points,
            PCFieldKey.CLOUD_LABEL: int(label[index]),  # type: ignore
        }

    def iterate_data(
        self, split: str, num_workers: int | None = 1
    ) -> Iterable[dict[str, str | np.ndarray]]:
        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}")

        return _DataIterator(
            self,
            split=split,
            num_workers=num_workers,
        )


class _DataIterator:
    def __init__(
        self,
        builder: ScanObjectNNBuilder,
        split: str,
        num_workers: int | None = 1,
    ):
        self.builder = builder
        self.num_workers = num_workers or 0

        zip_file = zipfile.ZipFile(builder.data_path, "r")
        file = zip_file.open(builder.H5_FILE_MAP[split])
        h5f = h5py.File(file, "r")

        self.indices = np.arange(len(h5f["data"]))  # type: ignore
        self.h5f_path = builder.H5_FILE_MAP[split]
        self.iter_args = [(self.h5f_path, index) for index in self.indices]

        self.results_len = len(self.indices)  # type: ignore

    def __iter__(self) -> Iterator[dict[str, str | np.ndarray]]:
        if self.num_workers > 0:
            with Pool(self.num_workers) as pool:
                self.results = pool.imap_unordered(
                    self.builder.process_instance, self.iter_args, chunksize=4
                )
                for result in self.results:
                    yield result
        else:
            for arg in self.iter_args:
                yield self.builder.process_instance(arg)

    def __len__(self):
        return self.results_len
