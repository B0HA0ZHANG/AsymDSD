from abc import ABC, abstractmethod
from typing import Any

import torch
import torchmetrics

from ..components import EncoderBranch, NormalizationTransform
from ..components.common_types import PathLike
from ..components.utils import init_lazy_defaults, lengths_to_mask
from ..data import PCFieldKey, SupervisedPCDataModule
from ..defaults import DEFAULT_NORM_TRANSFORM
from ..loggers import get_default_logger
from .embedding_model import EmbeddingModel
from .point_encoder import (
    # DEFAULT_POINT_ENCODER,
    PatchPoints,
    PointEncoder,
    PointEncoderOutput,
)

logger = get_default_logger()


class BaseEmbeddingClassifier(EmbeddingModel, ABC):
    # TODO: Move the classifier specific parts to another ABC class.
    @init_lazy_defaults
    def __init__(
        self,
        point_encoder: PointEncoder | None = None,
        encoder_ckpt_path: PathLike | None = None,
        encoder_choice: EncoderBranch | str = EncoderBranch.TEACHER,
        map_avg_pooling: bool = True,
        map_max_pooling: bool = False,
        map_cls_token: bool = False,
        num_classes: int | None = None,
        norm_transform: NormalizationTransform | None = DEFAULT_NORM_TRANSFORM,
        normalize_embeddings: bool = True,
        classifier_name: str = "",
        log_mean_acc: bool = False,
        save_embeddings_path: PathLike | None = None,
        load_embeddings_path: PathLike | None = None,
        top_k_metrics: int | list[int] = [1, 3],
    ):
        super().__init__(
            point_encoder=point_encoder,
            encoder_ckpt_path=encoder_ckpt_path,
            encoder_choice=encoder_choice,
            extract_patch_embeddings=False,
            map_avg_pooling=map_avg_pooling,
            map_max_pooling=map_max_pooling,
            map_cls_token=map_cls_token,
            norm_transform=norm_transform,
            normalize_embeddings=normalize_embeddings,
            save_embeddings_path=save_embeddings_path,
            load_embeddings_path=load_embeddings_path,
        )

        self.num_classes = num_classes
        self.classifier_name = classifier_name

        self.labels = []

        self.log_mean_acc = log_mean_acc
        self.top_acc_metrics = torch.nn.ModuleDict()
        # Convert top_k_metrics to list if it's an integer
        self.topk = [top_k_metrics] if isinstance(top_k_metrics, int) else top_k_metrics

    def setup(
        self,
        stage: str | None = None,
        datamodule: SupervisedPCDataModule | None = None,
    ):
        super().setup(stage=stage)

        if self.num_classes is None:
            if datamodule:
                self.num_classes = datamodule.num_classes[PCFieldKey.CLOUD_LABEL]
            else:
                datamodule = self.trainer.datamodule  # type: ignore
                self.num_classes = datamodule.num_classes[PCFieldKey.CLOUD_LABEL]  # type: ignore

        self.benchmark_name = datamodule.name if datamodule.name != "" else "benchmark"  # type: ignore

        accuracy_kwargs = {
            "task": "multiclass",
            "num_classes": self.num_classes,
            "average": "micro",
        }

        # Initialize top_k accuracy metrics for each k in self.topk
        for k in self.topk:
            self.top_acc_metrics[str(k)] = torchmetrics.Accuracy(
                top_k=k, **accuracy_kwargs
            )

        if self.log_mean_acc:
            accuracy_kwargs["average"] = "macro"
            self.mean_acc = torchmetrics.Accuracy(top_k=1, **accuracy_kwargs)

    def reset(self):
        super().reset()
        self.labels = []
        for metric in self.top_acc_metrics.values():
            metric.reset()
        if self.log_mean_acc:
            self.mean_acc.reset()

    def extract_embeddings(
        self, batch: dict[str, Any], return_raw_output: bool = False
    ) -> tuple[torch.Tensor, PointEncoderOutput | None]:
        patch_points = PatchPoints(
            points=batch[PCFieldKey.POINTS],
            num_points=batch.get("num_points"),
            patches_idx=batch.get("patches_idx"),
            centers_idx=batch.get("centers_idx"),
        )

        points = patch_points.points
        num_points = patch_points.num_points

        mask = (
            lengths_to_mask(num_points, points.size(1))
            if num_points is not None
            else None
        )

        points = self.norm_transform(points, mask=mask)
        embeddings = self(patch_points, return_raw_output=return_raw_output)

        return embeddings

    def training_step(self, batch: dict[str, torch.Tensor]):
        super().training_step(batch)

        labels = batch[PCFieldKey.CLOUD_LABEL]
        self.labels.append(labels)  # type: ignore

    def finalize_embeddings(self) -> None:
        if not self._is_finalized:
            self.embeddings = torch.cat(self.embeddings)  # type: ignore
            self.labels = torch.cat(self.labels)  # type: ignore
            self.embeddings, self.labels = self.filter_finite_embeddings(
                self.embeddings,
                self.labels,
                stage="fit",
            )

            self._is_finalized = True
            if self.save_embeddings_path:
                self.save_embeddings(self.save_embeddings_path)

    def filter_finite_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        finite_mask = torch.isfinite(embeddings).all(dim=-1)
        if finite_mask.all():
            return embeddings, labels

        num_bad = (~finite_mask).sum().item()
        logger.warning(
            "Dropping %s non-finite embedding samples during %s for %s.",
            num_bad,
            stage,
            self.classifier_name,
        )
        embeddings = embeddings[finite_mask]
        if labels is not None:
            labels = labels[finite_mask]
        return embeddings, labels

    def save_embeddings(self, path: PathLike) -> None:
        if not self._is_finalized:
            raise ValueError("Embeddings must be finalized before saving.")
        logger.info(f"Saving embeddings to {path}")
        torch.save(
            {
                "embeddings": self.embeddings.cpu(),  # type: ignore
                "patch_embeddings": self.patch_embeddings.cpu()  # type: ignore
                if self.extract_patch_embeddings
                else None,
                "labels": self.labels.cpu(),  # type: ignore
            },
            path,
        )

    def load_embeddings(
        self, path: PathLike, data: dict[str, torch.Tensor] | None = None
    ) -> None:
        if data is None:
            logger.info(f"Loading embeddings from {path}")
            data = torch.load(path)
        self.embeddings = data["embeddings"].to(self.device)  # type: ignore
        self.labels = data["labels"].to(self.device)  # type: ignore

        self._is_finalized = True

    def on_validation_epoch_start(self):
        self.finalize_embeddings()
        self.fit_model()

    # def on_train_epoch_end(self) -> None:
    #     self.finalize_embeddings()
    #     self.fit_model()

    @abstractmethod
    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def predict_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def fit_model(self):
        pass

    @property
    def name(self) -> str:
        return self.classifier_name

    def on_validation_epoch_end(self) -> None:
        log_dict = {}

        # Log all topk metrics
        for k, metric in self.top_acc_metrics.items():
            log_dict[f"{self.benchmark_name}/val/{self.name}/top{k}_acc"] = (
                metric.compute()
            )

        if self.log_mean_acc:
            log_dict[f"{self.benchmark_name}/val/{self.name}/mean_acc"] = (
                self.mean_acc.compute()
            )

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
