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

        if reduction == "none":
            return per_patch
        if reduction == "mean":
            return per_patch.mean()

        raise ValueError(f"Unsupported reduction: {reduction}")
