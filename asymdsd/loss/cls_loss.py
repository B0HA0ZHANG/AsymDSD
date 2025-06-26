import torch
from torch import nn


# TODO: Merge to single loss.py (Large overlap now also)
class ClsLoss(nn.Module):
    def compute_target_probs(
        self, target_logits: torch.Tensor, teacher_temp: float = 1.0
    ) -> torch.Tensor:
        target_probs = nn.functional.softmax(target_logits / teacher_temp, dim=-1)
        return target_probs

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_probs: torch.Tensor,
        student_temp: float = 1.0,
    ) -> torch.Tensor:
        # Sharpening (if temp < 1)
        pred_logits = pred_logits / student_temp
        pred_logprobs = nn.functional.log_softmax(pred_logits, dim=-1)

        # Matrix multiplication of target_probs and pred_logprobs^T
        # This computes CE for every student logit with every teacher logit
        loss = -torch.einsum("bik,bjk->bij", target_probs, pred_logprobs).mean()
        # TODO: Consider multiplying by inverse eye matrix to ignore the diagonal
        # Now it doesn't ignore 'the diagonal' See KoLeo loss for how to do this.

        return loss


class ClsRegressionLoss(nn.Module):
    def __init__(
        self,
        beta: float | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.loss_fn = nn.SmoothL1Loss(beta=beta) if beta is not None else nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # pred: (B, D, F)
        # target: (B, C, F)

        # Repeat targets D times, and pred C times
        target = target.unsqueeze(1).expand(-1, pred.shape[1], -1, -1)
        pred = pred.unsqueeze(2).expand(-1, -1, target.shape[2], -1)

        loss = self.loss_fn(pred, target)
        return loss
