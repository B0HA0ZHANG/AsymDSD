import copy
from enum import StrEnum, auto
from typing import Any

import lightning as L
import torch
import torchmetrics
import torchmetrics.segmentation
from jsonargparse import lazy_instance
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from torch import nn

from ..components import *
from ..components.checkpointing_utils import load_module_from_checkpoint
from ..components.scheduling import Schedule
from ..components.utils import (
    init_lazy_defaults,
    lengths_to_mask,
    sequentialize_transform,
)
from ..data import PCFieldKey, SupervisedPCDataModule
from ..defaults import *
from ..layers import *
from ..layers.patchify import PatchPoints
from ..layers.tokenization import *
from ..loggers import get_default_logger
from ..metrics import ShapeNetPartMeanIoU as MeanIoU
from .point_encoder import (
    DEFAULT_POINT_ENCODER,
    PointEncoder,
)

logger = get_default_logger()

DEFAULT_CLS_HEAD_CONFIG = lazy_instance(
    MLPConfig,
    dims=[512, 256],
    dropout_p=0.5,
    norm_layer=TransposeBatchNorm1d,
    bias=False,
)


class ClassificationHeadType(StrEnum):
    LINEAR = auto()
    MLP = auto()
    # MLP_ATTN = auto()


class SemanticSegementationModel(L.LightningModule):
    DEFAULT_BATCH_SIZE = 32

    @init_lazy_defaults
    def __init__(
        self,
        point_encoder: PointEncoder = DEFAULT_POINT_ENCODER,
        encoder_ckpt_path: str | None = None,
        encoder_choice: EncoderBranch | str = EncoderBranch.TEACHER,
        freeze_encoder: bool | int = False,
        map_avg_pooling: bool = True,
        map_max_pooling: bool = True,
        map_cls_token: bool = True,
        extract_hidden_layers: list[int] = [3, 7, 11],
        segmentation_head_config: MLPConfig = DEFAULT_CLS_HEAD_CONFIG,
        upsampling_dim: int = 384,
        label_embed_dim: int = 64,
        num_classes: int | None = None,
        num_seg_classes: int | None = None,
        subsampling_transform: SubsamplingTransform | None = None,
        aug_transform: AugmentationTransform | None = DEFAULT_AUG_TRANSFORM,
        norm_transform: NormalizationTransform | None = DEFAULT_NORM_TRANSFORM,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_epochs: int | None = 100,
        max_steps: int | None = None,
        steps_per_epoch: int | None = None,
        optimizer: OptimizerSpec = DEFAULT_CLASSIFIER_OPTIMIZER,
        label_smoothing: float = 0.1,
        init_weight_scale: float = 0.02,
        soft_init_head: bool = False,
        classifier_name: str = "neural",
    ) -> None:
        super().__init__()
        self.max_epochs = max_epochs if max_epochs and max_epochs > 0 else None
        self.max_steps = max_steps if max_steps and max_steps > 0 else None
        if max_steps is None and max_epochs is None:
            raise ValueError("Either max_epochs or max_steps must be specified.")

        self.steps_per_epoch = steps_per_epoch

        if not (map_avg_pooling or map_max_pooling or map_cls_token):
            raise ValueError(
                "At least one of map_avg_pooling, map_max_pooling, or map_cls_token must be True"
            )

        if map_cls_token and point_encoder.cls_token is None:
            map_cls_token = False
            logger.warning(
                "map_cls_token is True, but encoder does not have a cls token. Setting map_cls_token to False."
            )

        self.map_avg_pooling = map_avg_pooling
        self.map_max_pooling = map_max_pooling
        self.map_cls_token = map_cls_token

        self.extract_hidden_layers = extract_hidden_layers

        self.mlp_head_config = segmentation_head_config
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes

        self.encoder_ckpt_path = encoder_ckpt_path
        self.classifier_name = classifier_name

        self.batch_size = batch_size
        self.optimizer_spec = copy.deepcopy(optimizer)
        self.init_weight_scale = init_weight_scale
        self.soft_init_head = soft_init_head

        self.subsampling_transform = subsampling_transform or IdentityPassThrough()
        self.aug_transform: nn.Module = (
            sequentialize_transform(aug_transform)
            if aug_transform
            else IdentityMultiArg()
        )
        self.norm_transform: nn.Module = norm_transform or IdentityMultiArg()

        self.point_encoder = point_encoder
        self.encoder_choice = encoder_choice

        # Freeze encoder
        if isinstance(freeze_encoder, bool):
            freeze_encoder = -1 if freeze_encoder else 0
        self.freeze_encoder = freeze_encoder

        # Requires encoder to have embed_dim
        self.embed_dim = self.point_encoder.embed_dim
        self.upsampling_dim = upsampling_dim
        self.label_embed_dim = label_embed_dim

        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.point_upsample = PointUpsampling(  # TODO: add init args.
            3 + self.embed_dim * len(self.extract_hidden_layers),
            self.upsampling_dim,
            self.upsampling_dim,
        )
        self.label_embedding: nn.Module = None  # type: ignore
        self.segmentation_head: nn.Module = None  # type: ignore

        self.top1_acc_train: torchmetrics.Metric = None  # type: ignore

        self.top1_acc_val: torchmetrics.Metric = None  # type: ignore
        self.mean_iou: MeanIoU = None  # type: ignore

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.schedules = {
            "lr": self.optimizer_spec.lr,
            "wd": self.optimizer_spec.wd,
        }

        self.loaded_from_checkpoint = False

    def init_weights(self):
        std = self.init_weight_scale

        def _init_weights(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if not self.encoder_ckpt_path:
            self.point_encoder.apply(_init_weights)
            if self.point_encoder.cls_token is not None:
                nn.init.trunc_normal_(self.point_encoder.cls_token, std=std)

        if self.soft_init_head:
            self.segmentation_head.apply(_init_weights)  # Can consider kaiming_normal

    def _init_segmentation_head(self, num_seg_classes: int):
        input_dim = (
            (
                int(self.map_cls_token)
                + int(self.map_avg_pooling) * len(self.extract_hidden_layers)
                + int(self.map_max_pooling) * len(self.extract_hidden_layers)
            )
            * self.embed_dim
            + self.label_embed_dim
            + self.upsampling_dim
        )

        cfg: MLPConfig = self.mlp_head_config  # type: ignore
        # self.segmentation_head = MLPVarLen(
        #     *([input_dim] + cfg.dims + [num_seg_classes]),
        #     norm_layer=cfg.norm_layer,
        #     act_layer=cfg.act_layer,
        #     dropout_p=cfg.dropout_p,
        #     bias=cfg.bias,
        # )
        norm_layer = cfg.norm_layer or nn.Identity
        self.segmentation_head = nn.Sequential(
            nn.Linear(input_dim, cfg.dims[0], bias=cfg.bias),
            norm_layer(cfg.dims[0]),
            cfg.act_layer(),
            nn.Dropout(cfg.dropout_p),
            nn.Linear(cfg.dims[0], cfg.dims[1], bias=cfg.bias),
            norm_layer(cfg.dims[1]),
            cfg.act_layer(),
            # No dropout before last layer # TODO: CHECK
            nn.Linear(cfg.dims[1], num_seg_classes, bias=True),
        )

    def _init_metrics(self, num_classes: int, num_seg_classes: int):
        accuracy_kwargs = {
            "task": "multiclass",
            "num_classes": num_seg_classes,
            "average": "micro",
        }

        self.top1_acc_train = torchmetrics.Accuracy(top_k=1, **accuracy_kwargs)
        self.top1_acc_val = torchmetrics.Accuracy(top_k=1, **accuracy_kwargs)
        self.mean_iou = MeanIoU(
            num_segmentation_classes=num_seg_classes,
            num_instance_classes=num_classes,
        )

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                experiment = logger.experiment
                experiment.define_metric(
                    f"{self.benchmark}/val/{self.classifier_name}/instance_mIoU",
                    summary="last,max",
                )
                experiment.define_metric(
                    f"{self.benchmark}/val/{self.classifier_name}/class_mIoU",
                    summary="last,max",
                )

    def setup(
        self,
        stage: str | None = None,
        datamodule: SupervisedPCDataModule | None = None,
    ):
        if (
            self.steps_per_epoch is None
            or self.num_classes is None
            or self.num_seg_classes is None
        ) and datamodule is None:
            try:
                datamodule = self.trainer.datamodule  # type: ignore
            except AttributeError:
                raise ValueError(
                    "steps_per_epoch and num_classes must be specified if not using 'PointCloudData'"
                )

        self.benchmark = datamodule.name if datamodule.name != "" else "benchmark"  # type: ignore

        if self.num_classes is None:
            self.num_classes = datamodule.num_classes[PCFieldKey.CLOUD_LABEL]  # type: ignore

        if self.num_seg_classes is None:
            self.num_seg_classes = datamodule.num_classes[PCFieldKey.SEMANTIC_LABELS]  # type: ignore

        # TODO: Test other labelembedding
        # self.label_embedding = nn.Sequential(
        #     nn.Linear(self.num_classes, self.label_embed_dim, bias=False),
        #     TransposeBatchNorm1d(self.label_embed_dim),
        #     nn.LeakyReLU(0.2),
        # )
        self.label_embedding = nn.Embedding(self.num_classes, self.label_embed_dim)
        self._init_segmentation_head(self.num_seg_classes)

        if stage == "fit":
            if self.steps_per_epoch is None:
                self.steps_per_epoch = datamodule.len_train_dataset // self.batch_size  # type: ignore

            real_schedule: list[Schedule] = [
                s for s in self.schedules.values() if isinstance(s, Schedule)
            ]

            for schedule in real_schedule:
                max_epochs = self.max_epochs or (self.max_steps / self.steps_per_epoch)  # type: ignore
                schedule.set_default_max_epochs(max_epochs)  # type: ignore
                schedule.set_steps_per_epoch(self.steps_per_epoch)

        if stage == "fit" or stage == "test":
            self._init_metrics(self.num_classes, self.num_seg_classes)

        if self.encoder_ckpt_path:
            load_module_from_checkpoint(
                self.encoder_ckpt_path,
                module=self.point_encoder,
                device=self.device,
                key_prefix=[
                    "point_encoder",
                    "encoder",
                    "_encoder",
                    f"{str(self.encoder_choice)}.point_encoder",
                ],
                replace_key_part={
                    "attn_module": "self_attn",
                    "ffn_module": "ffn",
                },
            )

    def forward(self, patch_points: PatchPoints, cloud_label: int) -> torch.Tensor:
        multi_patches = self.point_encoder.patchify(patch_points)
        points = patch_points.points
        centers = multi_patches.centers[0]

        tokens: Tokens = self.point_encoder.patch_embedding(multi_patches)
        x = tokens.embeddings
        pos_enc = tokens.pos_embeddings

        embedding = self.point_encoder.transformer_encoder_forward(
            x,
            pos_enc,
            return_hidden_states=True,
        )
        # Select hidden states from specified layers
        patch_embeddings = torch.cat(
            [
                self.layer_norm(embedding.hidden_states[i][:, 1:])  # type: ignore
                for i in self.extract_hidden_layers
            ],
            dim=-1,
        )  # Maybe take the mean.

        x = self.point_upsample(centers, patch_embeddings, points, points)

        label_embedding = self.label_embedding(cloud_label)

        patch_features = [label_embedding]
        if self.map_cls_token:
            patch_features.append(embedding.cls_features)
        if self.map_avg_pooling:
            patch_features.append(patch_embeddings.mean(dim=1))
        if self.map_max_pooling:
            patch_features.append(patch_embeddings.amax(dim=1))

        patch_features = torch.cat(patch_features, dim=-1)
        patch_features = patch_features.unsqueeze(1).expand(-1, x.size(1), -1)

        x = torch.cat((x, patch_features), dim=-1)

        x = self.segmentation_head(x)

        return x

    def on_train_epoch_start(self) -> None:
        if (
            self.freeze_encoder == -1
            or self.trainer.current_epoch < self.freeze_encoder
        ):
            self.point_encoder.freeze()
        else:
            self.point_encoder.unfreeze()

    def forward_full(
        self, batch: dict[str, Any], augment_data: bool = False
    ) -> torch.Tensor:
        patch_points = PatchPoints(
            points=batch[PCFieldKey.POINTS],
            num_points=batch.get("num_points"),
            patches_idx=batch.get("patches_idx"),
            centers_idx=batch.get("centers_idx"),
        )

        points = patch_points.points
        num_points = patch_points.num_points

        points, num_points = self.subsampling_transform(points, num_points)

        mask = (
            lengths_to_mask(num_points, points.size(1))
            if num_points is not None
            else None
        )

        if augment_data:
            points = self.aug_transform(points)
        points = self.norm_transform(points, mask=mask)

        patch_points.points = points
        patch_points.num_points = num_points
        cloud_label = batch[PCFieldKey.CLOUD_LABEL]

        pred_logits = self(patch_points, cloud_label)
        return pred_logits

    def training_step(self, batch: dict[str, Any], batch_idx: int | None = None):
        pred_logits = self.forward_full(batch, augment_data=True)
        target_indices = batch[PCFieldKey.SEMANTIC_LABELS].long()

        # Expects logits to be (B, C, N), where C is the number of classes
        pred_logits = pred_logits.transpose(1, 2)
        loss = self.ce_loss(pred_logits, target_indices)

        self.top1_acc_train(pred_logits, target_indices)

        return {"loss": loss}

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int | None = None,
        dataloader_idx: int = 0,
    ):
        pred_logits = self.forward_full(batch, augment_data=True)
        target_point_indices = batch[PCFieldKey.SEMANTIC_LABELS].long()
        target_cloud_indices = batch[PCFieldKey.CLOUD_LABEL]

        pred_point_indices = torch.argmax(pred_logits, dim=-1)
        pred_logits = pred_logits.transpose(1, 2)

        loss = self.ce_loss(pred_logits, target_point_indices)
        self.top1_acc_val(pred_point_indices, target_point_indices)

        self.mean_iou.update(pred_logits, target_point_indices, target_cloud_indices)

        return {
            "loss": loss,
            "pred_indices": pred_point_indices,
            "target_point_indices": target_point_indices,
            "target_cloud_indices": target_cloud_indices,
        }

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int | None = None,
        dataloader_idx: int = 0,
    ):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch: dict[str, Any]) -> Any:
        pred_logits = self.forward_full(batch, augment_data=True)
        return {"pred_indices": pred_logits.argmax(dim=1)}

    def on_fit_start(self) -> None:
        # This is after checkpoint loading
        if not self.loaded_from_checkpoint:
            self.init_weights()

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self.log(
            f"{self.benchmark}/train/{self.classifier_name}/loss",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.log_dict(
            {
                f"{self.benchmark}/train/{self.classifier_name}/top1_acc": self.top1_acc_train,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.log_dict(
            {
                f"{self.benchmark}/val/{self.classifier_name}/loss": outputs["loss"],
                f"{self.benchmark}/val/{self.classifier_name}/top1_acc": self.top1_acc_val,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        inst_mIoU, cls_mIoU = self.mean_iou.compute()
        self.log_dict(
            {
                f"{self.benchmark}/val/{self.classifier_name}/instance_mIoU": inst_mIoU,
                f"{self.benchmark}/val/{self.classifier_name}/class_mIoU": cls_mIoU,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.mean_iou.reset()

    def on_test_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # When the encoder is always frozen, don't save encoder.
        if self.freeze_encoder == -1:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith("point_encoder"):
                    del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.loaded_from_checkpoint = True

        if self.freeze_encoder == -1:
            # Always frozen, so must load from SSRL checkpoint
            # The checkpoint connector does not support unstrict loading,
            # therefore add keys from current state point_encoder.
            ckpt_state_dict: dict[str, Any] = checkpoint["state_dict"]
            point_encoder_state_dict = self.state_dict()

            for k in list(point_encoder_state_dict.keys()):
                if k.startswith("point_encoder"):
                    ckpt_state_dict[k] = point_encoder_state_dict[k]

        if self.voting_augmentations is not None:
            ckpt_state_dict: dict[str, Any] = checkpoint["state_dict"]
            va_state_dict = self.voting_augmentations.state_dict()
            for k in list(va_state_dict.keys()):
                ckpt_state_dict[f"voting_augmentations.{k}"] = va_state_dict[k]

    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, metric: Any | None
    ) -> None:
        # Needs to overwrite to support scheduler that is not LRScheduler
        if metric is None:
            scheduler.step()  # type: ignore[call-arg]
        else:
            scheduler.step(metric)  # Also works for wd_schedule

    def configure_optimizers(self):
        # TODO: Make util function for this.
        # lr_multiplier = self.batch_size / SemanticSegementationModel.DEFAULT_BATCH_SIZE

        if self.freeze_encoder == -1:
            # Only give classification head parameters as the encoder will remain frozen.
            parameters = self.segmentation_head.parameters()
        else:
            parameters = self.parameters()

        optimizer = self.optimizer_spec.get_optim(parameters, 1.0)
        lr_scheduler = self.optimizer_spec.get_lr_scheduler(optimizer)
        weight_decay_scheduler = self.optimizer_spec.get_wd_scheduler(optimizer)

        optimizers = [optimizer]
        schedules = []

        if lr_scheduler is not None:
            schedules.append(
                {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "name": "lr_schedule",
                }
            )

        if weight_decay_scheduler is not None:
            schedules.append(
                {
                    "scheduler": weight_decay_scheduler,
                    "interval": "step",
                    "name": "wd_schedule",
                }
            )

        return optimizers, schedules
