import pickle
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

import numpy as np

from asymdsd.components.common_types import PathLike
from asymdsd.data import (
    ClassLabels,
    DataField,
    DatasetBuilder,
    FieldType,
    PCFieldKey,
)

YIELD_TYPE = dict[str, int | str | np.ndarray]


class ModelNetFewShotBuilder(DatasetBuilder):
    DATA_FILE = "ModelNetFewshot.zip"
    SPLITS = ["train", "test"]
    FILE_FORMAT = "pkl"

    def __init__(
        self,
        data_path: PathLike | None = None,
        n_way: Literal[5, 10] = 10,
        n_shot: Literal[10, 20] = 20,
        fold: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = 0,
    ):
        if data_path is None:
            data_path = (
                Path(__file__).parent / "data" / ModelNetFewShotBuilder.DATA_FILE
            )

        self._set_info(
            name=f"ModelNet Few-Shot: {n_way} way - {n_shot} shot - fold {fold}",
            data_path=data_path,
            splits=ModelNetFewShotBuilder.SPLITS,
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
            class_labels={PCFieldKey.CLOUD_LABEL: ClassLabels(n_way)},
        )

        self.n_shot = n_shot
        self.n_way = n_way
        self.fold = fold

    def process_instance(
        self, idx: int, data: tuple[np.ndarray, int, np.int32]
    ) -> YIELD_TYPE:
        points, label, _ = data

        points = points[:, [0, 2, 1]] # Swap y and z axes

        return {
            "name": str(idx),
            PCFieldKey.POINTS: points,
            PCFieldKey.CLOUD_LABEL: label,
        }

    def iterate_data(
        self, split: str, num_workers: int | None = None
    ) -> Iterable[YIELD_TYPE]:
        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}")

        return _DataIterator(
            self,
            split=split,
        )


class _DataIterator:
    def __init__(
        self,
        builder: ModelNetFewShotBuilder,
        split: str,
    ):
        self.builder = builder

        path = (
            f"ModelNetFewshot/{builder.n_way}way_{builder.n_shot}shot/{builder.fold}.pkl"
        )
        with zipfile.ZipFile(builder.data_path, "r") as zip_file:
            data_binary = zip_file.read(path)
        self.data = pickle.loads(data_binary)[split]

        self.results_len = len(self.data)

    def __iter__(self) -> Iterator[YIELD_TYPE]:
        for i in range(len(self)):
            yield self.builder.process_instance(i, self.data[i])

    def __len__(self):
        return self.results_len
