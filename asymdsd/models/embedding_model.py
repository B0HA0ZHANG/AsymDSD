from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch import nn

from ..components import EncoderBranch, NormalizationTransform
from ..components.checkpointing_utils import load_module_from_checkpoint
from ..components.common_types import PathLike
from ..components.utils import init_lazy_defaults, lengths_to_mask
from ..data import PCFieldKey
from ..defaults import DEFAULT_NORM_TRANSFORM
from ..layers import IdentityMultiArg
from ..loggers import get_default_logger
from .point_encoder import (
    # DEFAULT_POINT_ENCODER,
    PatchPoints,
    PointEncoder,
    PointEncoderOutput,
)

logger = get_default_logger()


class EmbeddingModel(L.LightningModule):
    @init_lazy_defaults
    def __init__(
        self,
        point_encoder: PointEncoder | None = None,
        encoder_ckpt_path: PathLike | None = None,
        encoder_choice: EncoderBranch | str = EncoderBranch.TEACHER,
        extract_patch_embeddings: bool = False,
        map_avg_pooling: bool = True,
        map_max_pooling: bool = False,
        map_cls_token: bool = False,
        norm_transform: NormalizationTransform | None = DEFAULT_NORM_TRANSFORM,
        normalize_embeddings: bool = True,
        save_embeddings_path: PathLike | None = None,
        load_embeddings_path: PathLike | None = None,
    ):
        super().__init__()
        if not (map_avg_pooling or map_max_pooling or map_cls_token):
            raise ValueError(
                "At least one of map_avg_pooling, map_max_pooling, or map_cls_token must be True"
            )

        self.extract_patch_embeddings = extract_patch_embeddings

        self.map_avg_pooling = map_avg_pooling
        self.map_max_pooling = map_max_pooling
        self.map_cls_token = map_cls_token

        self.point_encoder: PointEncoder = None  # type: ignore
        if point_encoder:
            self.setup_point_encoder(point_encoder)

        self.encoder_ckpt_path = encoder_ckpt_path
        self.encoder_choice = encoder_choice

        self.norm_transform = norm_transform or IdentityMultiArg()
        self.normalize_embeddings = normalize_embeddings

        self.save_embeddings_path = save_embeddings_path
        if save_embeddings_path:
            self.save_embeddings_path = Path(save_embeddings_path)
            self.save_embeddings_path.parent.mkdir(parents=True, exist_ok=True)

        self.load_embeddings_path = load_embeddings_path
        if load_embeddings_path:
            self.load_embeddings_path = Path(load_embeddings_path)

        self._is_finalized = False
        self.automatic_optimization = False

        self.embeddings = []
        self.patch_embeddings = []

        self.hparams.update(
            {
                "map_avg_pooling": map_avg_pooling,
                "map_max_pooling": map_max_pooling,
                "map_cls_token": map_cls_token,
                "normalize_embeddings": normalize_embeddings,
            }
        )

    def setup_point_encoder(self, point_encoder: PointEncoder):
        self.point_encoder = point_encoder

        if self.map_cls_token and point_encoder.cls_token is None:
            self.map_cls_token = False
            logger.warning(
                "map_cls_token is True, but encoder does not have a cls token. Setting map_cls_token to False."
            )

    def setup(
        self,
        stage: str | None = None,
    ):
        if not self.point_encoder:
            raise ValueError("Point encoder must be set before calling setup.")

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

        if self.load_embeddings_path:
            self.load_embeddings(self.load_embeddings_path)

        self.point_encoder.eval()

    @property
    def is_finalized(self) -> bool:
        return self._is_finalized

    @torch.no_grad()
    def forward(
        self,
        patch_points: PatchPoints,
        *,
        return_raw_output: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, PointEncoderOutput | None]:
        embedding: PointEncoderOutput = self.point_encoder(patch_points, **kwargs)

        features = []
        if self.map_cls_token:
            features.append(embedding.cls_features)
        if self.map_avg_pooling:
            features.append(embedding.patch_features.mean(dim=1))
        if self.map_max_pooling:
            features.append(embedding.patch_features.amax(dim=1))

        x = torch.cat(features, dim=-1)

        if self.normalize_embeddings:
            # Use normalized embeddings.
            x = nn.functional.normalize(x, dim=-1)

        return (x, embedding if return_raw_output else None)

    def reset(self):
        self._is_finalized = False
        self.embeddings = []
        self.patch_embeddings = []

    def on_train_epoch_start(self):
        if self._is_finalized:
            self.reset()
        self.point_encoder.eval()

    def extract_embeddings(
        self, batch: dict[str, Any], *, return_raw_output: bool = False, **kwargs: Any
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
        embeddings = self(patch_points, return_raw_output=return_raw_output, **kwargs)

        return embeddings

    def training_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, PointEncoderOutput | None]:
        if self._is_finalized:
            logger.warning(
                "Calling training_step after embeddings have been finalized.\n"
                "Resetting embeddings..."
            )
            self.reset()
        output = self.extract_embeddings(
            batch, return_raw_output=self.extract_patch_embeddings
        )
        self.embeddings.append(output[0])  # type: ignore
        if self.extract_patch_embeddings:
            self.patch_embeddings.append(output[1].patch_features)  # type: ignore
        return output

    def finalize_embeddings(self) -> None:
        if not self._is_finalized:
            self.embeddings = torch.cat(self.embeddings)  # type: ignore
            if self.extract_patch_embeddings:
                self.patch_embeddings = torch.cat(self.patch_embeddings)  # type: ignore

            self._is_finalized = True
            if self.save_embeddings_path:
                self.save_embeddings(self.save_embeddings_path)

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
            },
            path,
        )

    def load_embeddings(self, path: PathLike) -> None:
        logger.info(f"Loading embeddings from {path}")
        data = torch.load(path)
        self.embeddings = data["embeddings"].to(self.device)
        if self.extract_patch_embeddings:
            self.patch_embeddings = data["patch_embeddings"].to(self.device)

        self._is_finalized = True

    def on_train_epoch_end(self) -> None:
        self.finalize_embeddings()

    def configure_optimizers(self):
        return None
