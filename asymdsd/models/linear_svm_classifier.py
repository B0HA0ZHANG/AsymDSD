from typing import Any

import torch
from sklearn.svm import LinearSVC

from ..components import *
from ..data import PCFieldKey
from .base_embedding_classifier import BaseEmbeddingClassifier


class LinearSVMClassifier(BaseEmbeddingClassifier):
    CLASSIFIER_NAME = "linear_svm"

    def __init__(self, classifier_name=CLASSIFIER_NAME, **kwargs):
        super().__init__(classifier_name=classifier_name, **kwargs)
        self.svm = LinearSVC()

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        target_embeddings = self.extract_embeddings(batch)[0]
        val_labels: torch.Tensor = batch[PCFieldKey.CLOUD_LABEL]
        target_embeddings, filtered_labels = self.filter_finite_embeddings(
            target_embeddings,
            val_labels,
            stage="validation",
        )
        if target_embeddings.numel() == 0:
            return {"pred_indices": torch.empty(0, device=val_labels.device, dtype=torch.long), "target_indices": torch.empty(0, device=val_labels.device, dtype=torch.long)}
        val_labels = filtered_labels  # type: ignore[assignment]

        scores = self.svm.decision_function(target_embeddings.cpu().numpy())
        scores = torch.tensor(scores, device=val_labels.device)

        # Update all topk metrics
        for metric in self.top_acc_metrics.values():
            metric.update(scores, val_labels)

        if self.log_mean_acc:
            self.mean_acc.update(scores, val_labels)

        return {
            "pred_indices": scores.argmax(dim=1),
            "target_indices": val_labels,
        }

    def predict_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        embeddings = self.extract_embeddings(batch)[0]
        embeddings, _ = self.filter_finite_embeddings(
            embeddings,
            stage="predict",
        )
        if embeddings.numel() == 0:
            return {"pred_indices": torch.empty(0, device=batch[PCFieldKey.POINTS].device, dtype=torch.long)}

        predictions = self.svm.predict(embeddings.cpu().numpy())
        predictions = torch.tensor(predictions, device=embeddings.device)

        return {
            "pred_indices": predictions,
        }

    def fit_model(self):
        embeddings = self.embeddings.cpu().numpy()  # type: ignore
        labels = self.labels.cpu().numpy()  # type: ignore
        if embeddings.shape[0] == 0:
            raise RuntimeError("No finite training embeddings available for LinearSVM.")
        self.svm.fit(embeddings, labels)
