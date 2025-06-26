from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

from ..components.common_types import PathLike


class FieldType(StrEnum):
    STRING_LABEL = auto()
    INT_LABEL = auto()
    ARRAY = auto()


# TODO: Consider making datafields a dictionary from key to data_type (string, int, array),
#  and field_type (cloud_label, semantic etc.), and with an instantiation of data class
#  with additional properties such as classlabels.
@dataclass
class DataField:
    key: str
    key_type: FieldType | str


class PCFieldKey(StrEnum):
    POINTS = auto()
    FEATURES = auto()
    CLOUD_LABEL = auto()
    SEMANTIC_LABELS = auto()
    INSTANCE_LABELS = auto()


class ClassLabels:
    def __init__(self, labels: int | list[str]):
        if isinstance(labels, int):
            labels = [str(i) for i in range(labels)]
        self._label_names = labels
        self._num_classes = len(labels)

        enum_labels = enumerate(labels)
        self.str2int_dict = {label: i for i, label in enum_labels}

    @property
    def label_names(self) -> list[str]:
        return self._label_names

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def int2str(self, label_idx: int) -> str:
        return self.label_names[label_idx]

    def str2int(self, label_str: str) -> int:
        return self.str2int_dict[label_str]


class DatasetBuilder(ABC):
    def _set_info(
        self,
        name: str,
        data_path: PathLike,
        splits: list[str],
        data_fields: list[DataField],
        class_labels: dict[str, ClassLabels] | None = None,
    ):
        self._data_path = Path(data_path).expanduser().resolve()
        self._name = name
        self._splits = splits
        self._data_fields = data_fields
        self._class_labels = class_labels

    @abstractmethod
    def iterate_data(
        self, split: str, num_workers: int | None = None
    ) -> Iterable[dict[str, Any] | None]:
        pass

    def build(self, dataset_save_path: PathLike, num_workers: int | None = None):
        raise NotImplementedError

    @property
    def data_path(self) -> Path:
        return self._data_path

    @property
    def name(self) -> str:
        return self._name

    @property
    def splits(self) -> list[str]:
        return self._splits

    @property
    def data_fields(self) -> list[DataField]:
        return self._data_fields

    @property
    def class_labels(self) -> dict[str, ClassLabels] | None:
        return self._class_labels
