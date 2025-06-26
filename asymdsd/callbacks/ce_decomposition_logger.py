from typing import Any, Mapping

import lightning as L
import torch
from torch import nn

from asymdsd import AsymDSD


class CrossEntropyDecompositionLogger(L.Callback):
    def __init__(self, eps: float = 1e-7) -> None:
        self.eps = eps

    def decomposition(
        self,
        cross_entropy: torch.Tensor,
        target_logits: torch.Tensor,
        teacher_temp: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_probs = (
            nn.functional.softmax(target_logits / teacher_temp, dim=-1) + self.eps
        )

        target_posterior_entropy = (
            -(target_probs * target_probs.log()).sum(dim=-1).mean()
        )

        mean_target_probs = target_probs.mean(dim=tuple(range(target_probs.ndim - 1)))
        target_marginal_entropy = -(mean_target_probs * mean_target_probs.log()).sum()

        kl_div = cross_entropy - target_posterior_entropy

        return target_posterior_entropy, target_marginal_entropy, kl_div

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: AsymDSD,
        outputs: Mapping[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        if pl_module.mode.do_cls:
            cls_posterior_entropy, cls_marginal_entropy, cls_kl_div = (
                self.decomposition(
                    outputs["cls_loss"],
                    outputs["cls_targets"],
                    pl_module.scheduler.value["cls_teacher_temp"],
                )
            )
            pl_module.log_dict(
                {
                    "cls/target_entropy": cls_posterior_entropy,
                    "cls/marginal_entropy": cls_marginal_entropy,
                    "cls/kl_div": cls_kl_div,
                    "cls/cross_entropy": outputs["cls_loss"],
                },
                on_step=True,
            )
        if pl_module.mode.do_mask and not pl_module.disable_projection:
            patch_entropy, patch_marginal_entropy, patch_kl_div = self.decomposition(
                outputs["patch_loss"],
                outputs["patch_targets"],
                pl_module.scheduler.value["patch_teacher_temp"],
            )
            pl_module.log_dict(
                {
                    "patch/target_entropy": patch_entropy,
                    "patch/marginal_entropy": patch_marginal_entropy,
                    "patch/kl_div": patch_kl_div,
                    "patch/cross_entropy": outputs["patch_loss"],
                },
                on_step=True,
            )
