from typing import Any, Iterable

import lightning as L
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from asymdsd import AsymDSD
from asymdsd.components import EncoderBranch
from asymdsd.data import SupervisedPCDataModule
from asymdsd.models import PointEncoder
from asymdsd.models.base_embedding_classifier import BaseEmbeddingClassifier


class EmbeddingClassifierEval(L.Callback):
    def __init__(
        self,
        classifier: BaseEmbeddingClassifier | list[BaseEmbeddingClassifier],
        datamodule: SupervisedPCDataModule,
        eval_run_interval: int | list[int] = 1,
        encoder_choice: EncoderBranch | str = EncoderBranch.TEACHER,
        limit_train_batches: int | None = None,
        callbacks: list[L.Callback] | None = None,
        pre_empty_cache: bool = False,
        eval_name_prefix: str | None = None,
    ) -> None:
        super().__init__()
        self.eval_run_interval = eval_run_interval
        self.eval_name_prefix = (
            f"{eval_name_prefix}_" if eval_name_prefix is not None else ""
        )

        if isinstance(encoder_choice, str):
            encoder_choice = EncoderBranch(encoder_choice)
        self.encoder_choice = encoder_choice

        self.limit_train_batches = limit_train_batches
        self.callbacks = callbacks
        self.empty_cache = pre_empty_cache

        self.fabric = L.Fabric()

        self.trainer: L.Trainer = None  # type: ignore
        self._datamodule = datamodule
        if not isinstance(classifier, list):
            classifier = [classifier]
        self.classifiers = classifier

    def setup(
        self, trainer: L.Trainer, pl_module: AsymDSD, stage: str | None = None
    ) -> None:
        asymdsd = pl_module

        self._datamodule.prepare_data()
        self._datamodule.setup(stage=stage)  # type: ignore
        self.benchmark_name = (
            self._datamodule.name if self._datamodule.name != "" else "benchmark"
        )

        train_dataloader = self._datamodule.train_dataloader()
        val_dataloader = self._datamodule.val_dataloader()
        (
            self.train_dataloader,
            self.val_dataloader,
        ) = self.fabric.setup_dataloaders(train_dataloader, val_dataloader)  # type: ignore

        # No (deep)copy is made
        point_encoder = self._get_encoder(asymdsd)

        norm_transform = asymdsd.norm_transform

        for classifier in self.classifiers:
            classifier.setup_point_encoder(point_encoder)
            classifier.norm_transform = norm_transform
            classifier.benchmark_name = self.benchmark_name
            classifier.setup(datamodule=self._datamodule)
            self.fabric.setup_module(classifier)

        self._setup_callbacks(trainer)
        self._save_hparams(trainer)

    def _save_hparams(self, trainer: L.Trainer):
        trainer.logger.log_hyperparams(  # type: ignore
            self.hparams
        )

    @property
    def hparams(self):
        return {
            f"{self.eval_name_prefix}{classifier.name}_{self.benchmark_name}": {
                "eval_name": self.benchmark_name,
                "encoder_choice": self.encoder_choice,
                "limit_train_batches": self.limit_train_batches,
                **classifier.hparams,
            }
            for classifier in self.classifiers
        }

    def _get_encoder(self, asymdsd_module: AsymDSD) -> PointEncoder:  # type: ignore
        if self.encoder_choice == EncoderBranch.TEACHER:
            return asymdsd_module.teacher.point_encoder
        elif self.encoder_choice == EncoderBranch.STUDENT:
            return asymdsd_module.student.point_encoder

    def _wrap_progress_bar(
        self, dataloader: DataLoader, desc: str | None = None
    ) -> Iterable:
        return tqdm(dataloader, leave=False, desc=desc)

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: AsymDSD) -> None:
        validation_epoch = pl_module.validation_epoch
        if isinstance(self.eval_run_interval, list):
            if validation_epoch in self.eval_run_interval:
                self._run_evaluation(trainer, pl_module)
        else:
            if validation_epoch % self.eval_run_interval == self.eval_run_interval - 1:
                self._run_evaluation(trainer, pl_module)

    def _run_evaluation(self, trainer: L.Trainer, pl_module: AsymDSD) -> None:
        if self.empty_cache:
            torch.cuda.empty_cache()

        for classifier in self.classifiers:
            # TODO: No need to iterate, reuse embeddings (However needs to call on_train_epoch_end)
            self._fit(classifier)
            self._evaluate(classifier)
            self._log_metrics(classifier, pl_module)
            classifier.reset()

        if self.empty_cache:
            torch.cuda.empty_cache()

    def _fit(self, classifier: BaseEmbeddingClassifier):
        classifier.on_train_epoch_start()
        train_dataloader = self._wrap_progress_bar(
            self.train_dataloader,
            desc=f"Fitting {self.eval_name_prefix}{classifier.name} on {self.benchmark_name}",
        )
        for batch_idx, batch in enumerate(train_dataloader):
            if (
                self.limit_train_batches is not None
                and batch_idx >= self.limit_train_batches
            ):
                break
            classifier.training_step(batch)
        classifier.on_train_epoch_end()

    def _evaluate(self, classifier: BaseEmbeddingClassifier):
        classifier.on_validation_epoch_start()
        self._on_validation_start_callbacks(self.trainer)
        val_dataloader = self._wrap_progress_bar(
            self.val_dataloader,
            desc=f"Evaluating {self.eval_name_prefix}{classifier.name} on {self.benchmark_name}",
        )
        for batch_idx, batch in enumerate(val_dataloader):
            outputs = classifier.validation_step(batch)
            self._on_validation_batch_end_callbacks(
                self.trainer, classifier, outputs, batch, batch_idx
            )
        self._on_validation_end_callbacks(self.trainer, classifier)

    def _log_metrics(self, classifier: BaseEmbeddingClassifier, pl_module: AsymDSD):
        log_dict = {}

        # Log all topk metrics
        for k, metric in classifier.top_acc_metrics.items():
            log_dict[
                f"{self.benchmark_name}/val/{self.eval_name_prefix}{classifier.name}/top{k}_acc"
            ] = metric.compute()

        pl_module.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    # def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: AsymDSD) -> None:
    #     if trainer.current_epoch % self.eval_run_interval == self.eval_run_interval - 1:
    #         for classifier in self.classifiers:
    #             pl_module.log_dict(
    #                 {
    #                     f"{self.benchmark_name}/val/{self.eval_name_prefix}{classifier.name}/top1_acc": classifier.top1_acc.compute(),
    #                     f"{self.benchmark_name}/val/{self.eval_name_prefix}{classifier.name}/top3_acc": classifier.top3_acc.compute(),
    #                 },
    #                 on_step=False,
    #                 on_epoch=True,
    #                 prog_bar=True,
    #             )
    #     for classifier in self.classifiers:
    #         classifier.reset()

    def _setup_callbacks(self, trainer: L.Trainer):
        if self.callbacks is not None:
            for callback in self.callbacks:
                for classifier in self.classifiers:
                    callback.setup(trainer, classifier, stage="fit")

    def _on_validation_start_callbacks(self, trainer: L.Trainer):
        if self.callbacks is not None:
            for callback in self.callbacks:
                for classifier in self.classifiers:
                    callback.on_validation_start(self, classifier)  # type: ignore

    def _on_validation_batch_end_callbacks(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.callbacks is not None:
            for callback in self.callbacks:
                for classifier in self.classifiers:
                    callback.on_validation_batch_end(
                        trainer, classifier, outputs, batch, batch_idx
                    )

    def _on_validation_end_callbacks(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ):
        if self.callbacks is not None:
            for callback in self.callbacks:
                for classifier in self.classifiers:
                    callback.on_validation_end(trainer, classifier)

    @property
    def datamodule(self):
        return self._datamodule
