import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class PatchLoss(nn.Module):
    def forward(
        self,
        pred_logits: torch.Tensor,
        target_logits: torch.Tensor,
        teacher_temp: float = 1.0,
        student_temp: float = 1.0,
        reduction: str = "mean",
    ) -> torch.Tensor:
        # Sharpening (if temp < 1)
        pred_logits = pred_logits / student_temp
        target_logits = target_logits / teacher_temp

        pred_logprobs = nn.functional.log_softmax(pred_logits, dim=-1)
        target_probs = nn.functional.softmax(target_logits, dim=-1)

        per_patch = -(target_probs * pred_logprobs).sum(dim=-1)

        if reduction == "none":
            return per_patch
        if reduction == "mean":
            return per_patch.mean()

        raise ValueError(f"Unsupported reduction: {reduction}")


class MemEfficientPatchLoss(nn.Module):
    def temperatured_log_softmax(
        self, logits: torch.Tensor, temp: float
    ) -> torch.Tensor:
        return nn.functional.log_softmax(logits * (1 / temp), dim=-1)

    def temperatured_softmax(self, logits: torch.Tensor, temp: float) -> torch.Tensor:
        return nn.functional.softmax(logits * (1 / temp), dim=-1)

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_logits: torch.Tensor,
        teacher_temp: float = 1.0,
        student_temp: float = 1.0,
    ) -> torch.Tensor:
        pred_logprobs: torch.Tensor = checkpoint(  # type: ignore
            self.temperatured_log_softmax, pred_logits, student_temp
        )
        target_probs: torch.Tensor = checkpoint(  # type: ignore
            self.temperatured_softmax, target_logits, teacher_temp
        )

        loss = -(target_probs * pred_logprobs).sum(dim=-1).mean()

        return loss
