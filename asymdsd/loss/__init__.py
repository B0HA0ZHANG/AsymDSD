from .cls_loss import ClsLoss, ClsRegressionLoss
from .koleo_loss import KoLeoLoss
from .mean_entropy import MeanEntropyLoss
from .patch_loss import MemEfficientPatchLoss, PatchLoss

__all__ = [
    "ClsLoss",
    "ClsRegressionLoss",
    "KoLeoLoss",
    "MeanEntropyLoss",
    "PatchLoss",
    "MemEfficientPatchLoss",
]
