from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_gather, knn_points
from torch import nn

from ..components import FactoryConfig


@dataclass
class LocalRelationDistillLossConfig(FactoryConfig):
    num_neighbors: int = 8
    beta: float = 0.5

    @property
    def CLS(self):
        return LocalRelationDistillLoss

    def instantiate(self) -> "LocalRelationDistillLoss":
        return self.CLS(
            num_neighbors=self.num_neighbors,
            beta=self.beta,
        )


class LocalRelationDistillLoss(nn.Module):
    def __init__(self, num_neighbors: int = 8, beta: float = 0.5) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.beta = beta
        self.last_stats: dict[str, torch.Tensor] | None = None

    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
        centers: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if centers.size(1) <= 1:
            loss = torch.zeros(
                centers.size(0),
                centers.size(1),
                device=centers.device,
                dtype=student_emb.dtype,
            )
            return loss if reduction == "none" else loss.mean()

        k = min(self.num_neighbors + 1, centers.size(1))
        knn = knn_points(centers, centers, K=k, return_sorted=True)
        idx = knn.idx[:, :, 1:]

        student_nb = knn_gather(student_emb, idx)
        teacher_nb = knn_gather(teacher_emb, idx)

        student_rel = F.cosine_similarity(student_emb.unsqueeze(2), student_nb, dim=-1)
        teacher_rel = F.cosine_similarity(teacher_emb.unsqueeze(2), teacher_nb, dim=-1)

        per_patch = F.smooth_l1_loss(
            student_rel,
            teacher_rel.detach(),
            reduction="none",
            beta=self.beta,
        ).mean(dim=-1)
        batch_mean = per_patch.mean().detach()
        zero = batch_mean.new_zeros(())
        self.last_stats = {
            "relation_pos_loss": batch_mean,
            "relation_margin_loss": zero,
            "relation_teacher_margin": zero,
            "relation_student_margin": zero,
        }

        if reduction == "none":
            return per_patch
        if reduction == "mean":
            return per_patch.mean()

        raise ValueError(f"Unsupported reduction: {reduction}")


@dataclass
class DiscriminativeRelationDistillLossConfig(FactoryConfig):
    num_neighbors: int = 8
    beta: float = 0.5
    min_margin: float = 0.05
    num_hard_negatives: int = 4
    pos_weight: float = 1.0
    margin_weight: float = 1.0

    @property
    def CLS(self):
        return DiscriminativeRelationDistillLoss

    def instantiate(self) -> "DiscriminativeRelationDistillLoss":
        return self.CLS(
            num_neighbors=self.num_neighbors,
            beta=self.beta,
            min_margin=self.min_margin,
            num_hard_negatives=self.num_hard_negatives,
            pos_weight=self.pos_weight,
            margin_weight=self.margin_weight,
        )


class DiscriminativeRelationDistillLoss(nn.Module):
    def __init__(
        self,
        num_neighbors: int = 8,
        beta: float = 0.5,
        min_margin: float = 0.05,
        num_hard_negatives: int = 4,
        pos_weight: float = 1.0,
        margin_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.beta = beta
        self.min_margin = min_margin
        self.num_hard_negatives = num_hard_negatives
        self.pos_weight = pos_weight
        self.margin_weight = margin_weight
        self.last_stats: dict[str, torch.Tensor] | None = None

    def forward(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
        centers: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        batch_size, num_tokens, _ = student_emb.shape
        if num_tokens <= 1:
            loss = torch.zeros(
                batch_size,
                num_tokens,
                device=centers.device,
                dtype=student_emb.dtype,
            )
            zero = loss.mean().detach()
            self.last_stats = {
                "relation_pos_loss": zero,
                "relation_margin_loss": zero,
                "relation_teacher_margin": zero,
                "relation_student_margin": zero,
            }
            return loss if reduction == "none" else loss.mean()

        k = min(self.num_neighbors + 1, num_tokens)
        knn = knn_points(centers, centers, K=k, return_sorted=True)
        pos_idx = knn.idx[:, :, 1:]

        student_norm = F.normalize(student_emb, dim=-1)
        teacher_norm = F.normalize(teacher_emb.detach(), dim=-1)
        student_sim = torch.bmm(student_norm, student_norm.transpose(1, 2))
        teacher_sim = torch.bmm(teacher_norm, teacher_norm.transpose(1, 2))

        student_pos = student_sim.gather(2, pos_idx)
        teacher_pos = teacher_sim.gather(2, pos_idx)
        pos_loss = F.smooth_l1_loss(
            student_pos,
            teacher_pos,
            reduction="none",
            beta=self.beta,
        ).mean(dim=-1)

        eye = torch.eye(num_tokens, dtype=torch.bool, device=centers.device).unsqueeze(0)
        pos_mask = torch.zeros(
            batch_size,
            num_tokens,
            num_tokens,
            dtype=torch.bool,
            device=centers.device,
        )
        pos_mask.scatter_(2, pos_idx, True)
        neg_mask = ~(pos_mask | eye)

        topk = min(self.num_hard_negatives, num_tokens - 1)
        if topk > 0:
            teacher_neg_scores = teacher_sim.masked_fill(~neg_mask, float("-inf"))
            teacher_hard_neg, hard_idx = teacher_neg_scores.topk(k=topk, dim=-1)
            valid_neg = torch.isfinite(teacher_hard_neg)
            student_hard_neg = student_sim.gather(2, hard_idx)

            teacher_hard_neg = torch.where(
                valid_neg, teacher_hard_neg, torch.zeros_like(teacher_hard_neg)
            )
            student_hard_neg = torch.where(
                valid_neg, student_hard_neg, torch.zeros_like(student_hard_neg)
            )

            valid_neg_count = valid_neg.sum(dim=-1)
            valid_neg_denom = valid_neg_count.clamp_min(1).to(student_emb.dtype)
            has_neg = valid_neg_count > 0

            teacher_pos_mean = teacher_pos.mean(dim=-1)
            student_pos_mean = student_pos.mean(dim=-1)
            teacher_neg_mean = teacher_hard_neg.sum(dim=-1) / valid_neg_denom
            student_neg_mean = student_hard_neg.sum(dim=-1) / valid_neg_denom

            teacher_margin = teacher_pos_mean - teacher_neg_mean
            student_margin = student_pos_mean - student_neg_mean
            target_margin = teacher_margin.clamp_min(self.min_margin)
            margin_loss = F.relu(target_margin - student_margin)

            teacher_margin = torch.where(
                has_neg, teacher_margin, torch.zeros_like(teacher_margin)
            )
            student_margin = torch.where(
                has_neg, student_margin, torch.zeros_like(student_margin)
            )
            margin_loss = torch.where(
                has_neg, margin_loss, torch.zeros_like(margin_loss)
            )
        else:
            teacher_margin = torch.zeros(
                batch_size,
                num_tokens,
                device=centers.device,
                dtype=student_emb.dtype,
            )
            student_margin = torch.zeros_like(teacher_margin)
            margin_loss = torch.zeros_like(teacher_margin)

        per_patch = self.pos_weight * pos_loss + self.margin_weight * margin_loss
        self.last_stats = {
            "relation_pos_loss": pos_loss.mean().detach(),
            "relation_margin_loss": margin_loss.mean().detach(),
            "relation_teacher_margin": teacher_margin.mean().detach(),
            "relation_student_margin": student_margin.mean().detach(),
        }

        if reduction == "none":
            return per_patch
        if reduction == "mean":
            return per_patch.mean()

        raise ValueError(f"Unsupported reduction: {reduction}")
