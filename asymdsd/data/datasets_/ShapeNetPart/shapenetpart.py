import json
import tarfile
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

from .synset_map import SYNSET_MAP


@lru_cache(1)
def open_tarfile_buffer(data_path):
    return tarfile.TarFile(data_path, "r")


class ShapeNetPartBuilder(DatasetBuilder):
    DATA_FILE = "shapenetcore_partanno_segmentation_benchmark_v0_normal.tar"
    SPLITS = ["train", "val", "test"]
    FILE_FORMAT = "txt"

    SYNSET_MAP = SYNSET_MAP
    NUM_SEMANTIC_CLASSES = 50

    def __init__(
        self,
        data_path: PathLike | None = None,
        num_pre_sample_points: int = 16384,
        seed: int | None = None,
    ):
        if data_path is None:
            data_path = Path(__file__).parent / "data" / ShapeNetPartBuilder.DATA_FILE

        label_names = list(SYNSET_MAP.values())

        self._set_info(
            name="ShapeNetPart",
            data_path=data_path,
            splits=ShapeNetPartBuilder.SPLITS,
            data_fields=[
                DataField(
                    key=PCFieldKey.POINTS,
                    key_type=FieldType.ARRAY,
                ),
                DataField(
                    key=PCFieldKey.CLOUD_LABEL,
                    key_type=FieldType.STRING_LABEL,
                ),
                DataField(
                    key=PCFieldKey.SEMANTIC_LABELS,
                    key_type=FieldType.ARRAY,
                ),
            ],
            class_labels={
                PCFieldKey.CLOUD_LABEL: ClassLabels(label_names),
                PCFieldKey.SEMANTIC_LABELS: ClassLabels(
                    ShapeNetPartBuilder.NUM_SEMANTIC_CLASSES
                ),
            },
        )

        self.num_pre_sample_points = num_pre_sample_points

    def process_instance(
        self, tar_member: tarfile.TarInfo
    ) -> dict[str, str | np.ndarray]:
        tar_file = open_tarfile_buffer(self.data_path)
        with tar_file.extractfile(tar_member) as file:
            data = np.loadtxt(file, delimiter=" ", dtype=np.float32)

        points = data[:, :3]
        semantic_labels = data[:, -1].astype(np.int32)

        # Rotate axes system such that z is up.
        points[:, :] = points[:, [2, 0, 1]]

        path_parts = tar_member.name.split("/")
        synset_id = path_parts[-2]

        name = f"{synset_id}_{path_parts[-1].removesuffix(f'.{ShapeNetPartBuilder.FILE_FORMAT}')}"
        label = ShapeNetPartBuilder.SYNSET_MAP[synset_id]

        return {
            "name": name,
            PCFieldKey.POINTS: points,
            PCFieldKey.SEMANTIC_LABELS: semantic_labels,
            PCFieldKey.CLOUD_LABEL: label,
        }

    def iterate_data(
        self, split: str, num_workers: int = 1
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
        builder: ShapeNetPartBuilder,
        split: str,
        num_workers: int | None = 1,
    ):
        self.builder = builder
        self.num_workers = num_workers or 0

        with tarfile.open(builder.data_path, "r") as tar:
            members = tar.getmembers()
            index = {member.name: member for member in members}

            # find tarinfo object for name that ends with shuffled_{split}_file_list.json
            split_file = [
                name
                for name in index
                if name.endswith(f"shuffled_{split}_file_list.json")
            ][0]
            with tar.extractfile(index[split_file]) as file:
                paths = json.load(file)

        replace_name = list(index.keys())[0]

        # Replace the file prefix before first / with the correct path
        paths = [f"{replace_name}/{path.split('/', 1)[1]}.txt" for path in paths]

        self.process_members = [index[path] for path in paths]

        self.results_len = len(self.process_members)

    def __iter__(self) -> Iterator[dict[str, str | np.ndarray]]:
        if self.num_workers > 0:
            with Pool(self.num_workers) as pool:
                self.results = pool.imap_unordered(
                    self.builder.process_instance, self.process_members, chunksize=4
                )
                for result in self.results:
                    yield result
        else:
            for path in self.process_members:
                yield self.builder.process_instance(path)

    def __len__(self):
        return self.results_len
