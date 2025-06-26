from pathlib import Path
from typing import Any, Mapping

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from sklearn import metrics

# from wandb.sklearn import utils as wandb_utils


class ConfusionMatrixLogger(L.Callback):
    def __init__(
        self,
        save_every_n_epochs: int = 1,
        figsize: tuple[float, float] = (8.0, 8.0),
        log_dir: str | None = None,
    ) -> None:
        self.interval = save_every_n_epochs
        self.epoch = 0
        self.figsize = figsize
        self.log_dir = Path(log_dir) if log_dir is not None else None

    def setup(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        stage: str,
    ) -> None:
        if not isinstance(trainer.logger, WandbLogger):
            raise ValueError("ConfusionMatrixLogger only works with WandbLogger")
        # wandb_utils.chart_limit = 5000

        self.experiment_name = trainer.logger.experiment.name

        if self.log_dir is not None:
            self.log_dir = self.log_dir / self.experiment_name
            self.log_dir.mkdir(parents=True)

    def on_validation_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if hasattr(trainer, "datamodule") and hasattr(
            trainer.datamodule,
            "label_names",  # type: ignore
        ):  # type: ignore
            self.label_names = trainer.datamodule.label_names["cloud_label"]  # type: ignore
        elif hasattr(trainer, "callbacks"):
            for callback in trainer.callbacks:  # type: ignore
                if hasattr(callback, "datamodule") and hasattr(
                    callback.datamodule, "label_names"
                ):
                    self.label_names = callback.datamodule.label_names["cloud_label"]
                    break
        else:
            raise ValueError("No label_names property found in datamodule.")

        self.y_pred = []
        self.y_true = []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        pred_indices: torch.Tensor = outputs["pred_indices"]
        target_indices: torch.Tensor = outputs["target_indices"]

        # Convert to numpy array and append to list
        self.y_pred.append(pred_indices.cpu().numpy())  # type: ignore
        self.y_true.append(target_indices.cpu().numpy())  # type: ignore

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self.epoch % self.interval != 0:
            self.epoch += 1
            return

        self.y_pred = np.concatenate(self.y_pred)
        self.y_true = np.concatenate(self.y_true)

        fig, ax = plt.subplots(figsize=self.figsize, layout="tight")

        # Separate confusion matrix and normalized confusion matrix
        # such that we can use normalized values for colors and absolute values for text
        cm = metrics.confusion_matrix(self.y_true, self.y_pred)
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # Use seaborn heatmap with row-wise normalized colors but absolute values as text
        sns.heatmap(
            cm_normalized,  # Normalized colors
            annot=cm,  # Absolute values
            fmt="d",  # Integer format for values
            cmap="Blues",  # Colormap
            linewidths=0.5,  # Line thickness between cells
            ax=ax,  # Axes
            xticklabels=self.label_names,  # Custom X-axis labels
            yticklabels=self.label_names,  # Custom Y-axis labels
            cbar=False,  # Disable colorbar
            annot_kws={"size": 8},
        )

        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)

        wandb.log({"confusion_matrix": wandb.Image(fig)})

        plt.close(fig)

        if self.log_dir is not None:
            np.save(self.log_dir / f"confusion_matrix_{self.epoch}.npy", cm)
            artifact = wandb.Artifact(
                f"confusion_matrix_{self.epoch}", type="confusion_matrix"
            )
            artifact.add_file(str(self.log_dir / f"confusion_matrix_{self.epoch}.npy"))

            wandb.log_artifact(artifact)

        self.epoch += 1

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.on_validation_start(trainer, pl_module)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.on_validation_end(trainer, pl_module)
