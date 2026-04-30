from .cls_loss import ClsLoss, ClsRegressionLoss
from .koleo_loss import KoLeoLoss
from .local_relation_distill_loss import (
    DiscriminativeRelationDistillLoss,
    DiscriminativeRelationDistillLossConfig,
    LocalRelationDistillLoss,
    LocalRelationDistillLossConfig,
)
from .mean_entropy import MeanEntropyLoss
from .patch_loss import MemEfficientPatchLoss, PatchLoss

__all__ = [
    "ClsLoss",
    "ClsRegressionLoss",
    "KoLeoLoss",
    "DiscriminativeRelationDistillLoss",
    "DiscriminativeRelationDistillLossConfig",
    "LocalRelationDistillLoss",
    "LocalRelationDistillLossConfig",
    "MeanEntropyLoss",
    "PatchLoss",
    "MemEfficientPatchLoss",
]
