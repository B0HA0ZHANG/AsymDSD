from typing import Any, Iterable

import lightning as L
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from asymdsd import AsymDSD
from asymdsd.components import EncoderBranch
from asymdsd.components.utils import lengths_to_mask
from asymdsd.data import PCFieldKey, SupervisedPCDataModule
from asymdsd.layers.patchify import PatchPoints
from asymdsd.layers.tokenization import Tokens
from asymdsd.models import PointEncoder


class SemanticTokenClusteringEval(L.Callback):
    def __init__(
        self,
        datamodule: SupervisedPCDataModule,
        eval_run_interval: int | list[int] = 1,
        encoder_choice: EncoderBranch | str = EncoderBranch.TEACHER,
        num_clusters: int | None = None,
        max_tokens: int = 20000,
        limit_batches: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if isinstance(encoder_choice, str):
            encoder_choice = EncoderBranch(encoder_choice)
        self.encoder_choice = encoder_choice
        self.eval_run_interval = eval_run_interval
        self.num_clusters = num_clusters
        self.max_tokens = max_tokens
        self.limit_batches = limit_batches
        self.seed = seed
        self._datamodule = datamodule

    def setup(
        self,
        trainer: L.Trainer,
        pl_module: AsymDSD,
        stage: str | None = None,
    ) -> None:
        self._datamodule.prepare_data()
        self._datamodule.setup(stage=stage)  # type: ignore
        self.benchmark_name = (
            self._datamodule.name if self._datamodule.name != "" else "benchmark"
        )
        self.val_dataloader = self._datamodule.val_dataloader()

        num_classes = self._datamodule.num_classes.get(PCFieldKey.SEMANTIC_LABELS)
        self._num_semantic_classes = num_classes

    def on_validation_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: AsymDSD,
    ) -> None:
        validation_epoch = pl_module.validation_epoch
        if isinstance(self.eval_run_interval, list):
            should_eval = validation_epoch in self.eval_run_interval
        else:
            should_eval = validation_epoch % self.eval_run_interval == (
                self.eval_run_interval - 1
            )

        if should_eval:
            self._run_evaluation(pl_module)

    def _run_evaluation(self, pl_module: AsymDSD) -> None:
        encoder = self._get_encoder(pl_module)
        was_training = encoder.training
        encoder.eval()

        features, labels = self._collect_tokens(pl_module, encoder)
        if was_training:
            encoder.train()

        if features.size(0) < 2:
            return

        if self.max_tokens > 0 and features.size(0) > self.max_tokens:
            generator = torch.Generator().manual_seed(self.seed)
            idx = torch.randperm(features.size(0), generator=generator)[: self.max_tokens]
            features = features[idx]
            labels = labels[idx]

        metrics = self._cluster_metrics(features.numpy(), labels.numpy())
        pl_module.log_dict(
            {
                f"{self.benchmark_name}/val/semantic_token/{name}": value
                for name, value in metrics.items()
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def _get_encoder(self, asymdsd_module: AsymDSD) -> PointEncoder:
        if self.encoder_choice == EncoderBranch.TEACHER:
            return asymdsd_module.teacher.point_encoder
        if self.encoder_choice == EncoderBranch.STUDENT:
            return asymdsd_module.student.point_encoder
        raise ValueError(f"Unsupported encoder choice: {self.encoder_choice}")

    def _wrap_progress_bar(
        self,
        dataloader: DataLoader,
        desc: str | None = None,
    ) -> Iterable:
        return tqdm(dataloader, leave=False, desc=desc)

    def _collect_tokens(
        self,
        pl_module: AsymDSD,
        encoder: PointEncoder,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_features = []
        all_labels = []
        device = pl_module.device

        dataloader = self._wrap_progress_bar(
            self.val_dataloader,
            desc=f"Semantic token clustering on {self.benchmark_name}",
        )
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if self.limit_batches is not None and batch_idx >= self.limit_batches:
                    break
                batch = self._to_device(batch, device)
                patch_features = self._extract_patch_features(pl_module, encoder, batch)
                patch_labels = self._extract_patch_labels(batch)

                all_features.append(patch_features.flatten(0, 1).detach().cpu())
                all_labels.append(patch_labels.flatten(0, 1).detach().cpu())

        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    def _extract_patch_features(
        self,
        pl_module: AsymDSD,
        encoder: PointEncoder,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        points = batch[PCFieldKey.POINTS]
        num_points = batch.get("num_points")
        mask = (
            lengths_to_mask(num_points, points.size(1))
            if num_points is not None
            else None
        )
        points = pl_module.norm_transform(points, mask=mask)

        patch_points = PatchPoints(
            points=points,
            num_points=num_points,
            patches_idx=batch.get("patches_idx"),
            centers_idx=batch.get("centers_idx"),
        )
        multi_patches = encoder.patchify(patch_points)
        tokens: Tokens = encoder.patch_embedding(multi_patches)
        out = encoder.transformer_encoder_forward(
            tokens.embeddings,
            tokens.pos_embeddings,
            token_centers=tokens.centers,
            attn_bias_scale=pl_module.scheduler.value.get("relative_3d_bias_scale", 0.0),
        )
        return out.patch_features

    def _extract_patch_labels(self, batch: dict[str, Any]) -> torch.Tensor:
        semantic_labels = batch[PCFieldKey.SEMANTIC_LABELS].long()
        patches_idx = batch["patches_idx"][0]
        patch_point_labels = semantic_labels.gather(
            1,
            patches_idx.flatten(1).long(),
        ).reshape_as(patches_idx)
        return patch_point_labels.mode(dim=-1).values

    def _cluster_metrics(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float]:
        labels = labels.astype(np.int64)
        num_clusters = self.num_clusters or self._num_semantic_classes
        num_clusters = int(num_clusters or int(labels.max() + 1))
        num_clusters = min(num_clusters, features.shape[0])

        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            random_state=self.seed,
            batch_size=min(4096, max(256, features.shape[0])),
            n_init=3,
        )
        clusters = kmeans.fit_predict(features)

        return {
            "nmi": float(normalized_mutual_info_score(labels, clusters)),
            "ari": float(adjusted_rand_score(labels, clusters)),
            "purity": self._cluster_purity(labels, clusters, num_clusters),
            "hungarian_miou": self._hungarian_miou(labels, clusters, num_clusters),
        }

    def _contingency(
        self,
        labels: np.ndarray,
        clusters: np.ndarray,
        num_clusters: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        label_values = np.unique(labels)
        label_to_idx = {label: idx for idx, label in enumerate(label_values)}
        contingency = np.zeros((num_clusters, len(label_values)), dtype=np.int64)
        for label, cluster in zip(labels, clusters):
            contingency[cluster, label_to_idx[label]] += 1
        return contingency, label_values

    def _cluster_purity(
        self,
        labels: np.ndarray,
        clusters: np.ndarray,
        num_clusters: int,
    ) -> float:
        contingency, _ = self._contingency(labels, clusters, num_clusters)
        return float(contingency.max(axis=1).sum() / max(1, labels.shape[0]))

    def _hungarian_miou(
        self,
        labels: np.ndarray,
        clusters: np.ndarray,
        num_clusters: int,
    ) -> float:
        contingency, label_values = self._contingency(labels, clusters, num_clusters)
        cluster_idx, label_idx = linear_sum_assignment(-contingency)
        mapping = {cluster: label_values[label] for cluster, label in zip(cluster_idx, label_idx)}
        mapped = np.array([mapping.get(cluster, -1) for cluster in clusters])

        ious = []
        for label in label_values:
            pred_mask = mapped == label
            label_mask = labels == label
            union = np.logical_or(pred_mask, label_mask).sum()
            if union > 0:
                intersection = np.logical_and(pred_mask, label_mask).sum()
                ious.append(intersection / union)
        return float(np.mean(ious)) if ious else 0.0

    def _to_device(self, value: Any, device: torch.device) -> Any:
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: self._to_device(val, device) for key, val in value.items()}
        if isinstance(value, list):
            return [self._to_device(val, device) for val in value]
        return value
