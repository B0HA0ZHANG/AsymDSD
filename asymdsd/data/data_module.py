import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum

import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .dataset_builder import PCFieldKey
from .dataset_utils import get_dataset_key


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PREDICT = "predict"


class SupervisedDataModuleHooks(ABC):
    @property
    def num_classes(self) -> dict[str | PCFieldKey, int]:
        return {}

    @property
    def label_names(self) -> dict[str | PCFieldKey, list[str]]:
        return {}

    @property
    def label_int2str(self) -> dict[str | PCFieldKey, Callable[[int], str]]:
        return {}


class PointCloudDataModule(L.LightningDataModule):
    def __init__(
        self,
        name: str | None = None,
        batch_size: int = 32,
        num_workers_train: int = 0,
        num_workers_val_test: int = 0,
        num_workers_predict: int = 0,
        pin_memory: bool = False,
        seed: int | None = None,
    ):
        super().__init__()
        self._name = name
        self.batch_size = batch_size
        self.num_workers_train = num_workers_train
        self.num_workers_val_test = num_workers_val_test
        self.num_workers_predict = num_workers_predict

        self.dl_kwargs = {
            "batch_size": batch_size,
            "pin_memory": pin_memory,
        }

        self._dataset: dict[DatasetSplit, Dataset] = {}

        if seed is not None:
            self.seed = seed
        else:
            env_seed = os.environ.get("PL_GLOBAL_SEED")
            self.seed = int(env_seed) if env_seed is not None else None

        self.np_rng = np.random.default_rng(self.seed)

    @property
    def name(self) -> str:
        return self._name if self._name is not None else ""

    @property
    def dataset(self) -> dict[DatasetSplit, Dataset]:
        return self._dataset

    @dataset.setter
    def dataset(self, value: dict[DatasetSplit, Dataset]) -> None:
        self._dataset = value

    @property
    @abstractmethod
    def len_train_dataset(self) -> int | None:
        pass

    def train_dataloader(self, drop_last=True) -> DataLoader:
        dataset = self.dataset[DatasetSplit.TRAIN]
        # Shuffle is implemented by iterable dataset.
        # Train is expected to be present
        return DataLoader(
            self.dataset[DatasetSplit.TRAIN],
            drop_last=drop_last,
            shuffle=False if isinstance(dataset, IterableDataset) else True,
            num_workers=self.num_workers_train,
            persistent_workers=self.num_workers_train > 0,
            **self.dl_kwargs,
        )

    def val_dataloader(self) -> DataLoader | list:
        dataset_key = get_dataset_key(
            self.dataset, [DatasetSplit.VALIDATION, DatasetSplit.TEST]
        )
        if dataset_key is None:
            return []
        return DataLoader(
            self.dataset[dataset_key],
            drop_last=False,
            num_workers=self.num_workers_val_test,
            persistent_workers=self.num_workers_val_test > 0,
            **self.dl_kwargs,
        )

    def test_dataloader(self) -> DataLoader | list:
        dataset_key = get_dataset_key(
            self.dataset, [DatasetSplit.TEST, DatasetSplit.VALIDATION]
        )
        if dataset_key is None:
            return []
        return DataLoader(
            self.dataset[dataset_key],
            drop_last=False,
            num_workers=self.num_workers_val_test,
            persistent_workers=self.num_workers_val_test > 0,
            **self.dl_kwargs,
        )

    def predict_dataloader(self) -> DataLoader | None:
        dataset_key = get_dataset_key(self.dataset, [DatasetSplit.PREDICT])
        if dataset_key is None:
            return None
        return DataLoader(
            self.dataset[dataset_key],
            drop_last=False,
            num_workers=self.num_workers_predict,
            persistent_workers=self.num_workers_predict > 0,
            **self.dl_kwargs,
        )


class SupervisedPCDataModule(SupervisedDataModuleHooks, PointCloudDataModule):
    def __init__(
        self,
        name: str | None = None,
        batch_size: int = 32,
        num_workers_train: int = 0,
        num_workers_val_test: int = 0,
        num_workers_predict: int = 0,
        pin_memory: bool = False,
        seed: int | None = None,
    ):
        super().__init__(
            name=name,
            batch_size=batch_size,
            num_workers_train=num_workers_train,
            num_workers_val_test=num_workers_val_test,
            num_workers_predict=num_workers_predict,
            pin_memory=pin_memory,
            seed=seed,
        )

    pass  # For typing
