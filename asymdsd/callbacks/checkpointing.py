import re

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from ..components.common_types import PathLike


class DefaultTrainerCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        filename: str | None = None,
        dirpath: PathLike = "checkpoints",
        save_last: bool = False,
        monitor: str | None = None,
        mode: str = "max",
        save_on_fit_end: bool = False,
        **kwargs,
    ) -> None:
        self.experiment_name = ""
        self.base_name = ""
        self.monitor_name = ""

        if not filename:
            self.base_name = "epoch={epoch}-step={step}"
            if monitor is not None:
                self._set_monitor_name(monitor)
        else:
            self.base_name = filename

        self._update_filename()

        self.save_on_fit_end = save_on_fit_end

        super().__init__(
            dirpath,
            filename=self.filename,
            save_last=save_last,
            monitor=monitor,
            mode=mode,
            auto_insert_metric_name=False,
            **kwargs,
        )

    def _update_filename(self) -> None:
        parts = [self.experiment_name, self.base_name, self.monitor_name]
        self.filename = "-".join(part for part in parts if part)

    def _set_monitor_name(self, monitor: str) -> None:
        self.monitor_name = (
            f"{monitor.replace('/', '_')}"
            f"{ModelCheckpoint.CHECKPOINT_EQUALS_CHAR}"
            f"{{{monitor}:.3f}}"
        )
        self._update_filename()

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            self.experiment_name = logger.experiment.name
            self._update_filename()

    def _save_topk_checkpoint(
        self, trainer: "L.Trainer", monitor_candidates: dict[str, torch.Tensor]
    ) -> None:
        if self.save_top_k == 0:
            return

        if self.monitor is not None and "{..}" in self.monitor:
            pattern = re.escape(self.monitor).replace(r"\{\.\.\}", ".*")
            regex = re.compile(pattern)

            # Find matching keys in monitor_candidates
            for key in monitor_candidates:
                if regex.match(key):
                    self.monitor = key
                    break

            self._set_monitor_name(self.monitor)

        super()._save_topk_checkpoint(trainer, monitor_candidates)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Helpful if saving on train step interval.
        if not self.save_on_fit_end or self._should_skip_saving_checkpoint(trainer):
            return
        monitor_candidates = self._monitor_candidates(trainer)
        self._save_topk_checkpoint(trainer, monitor_candidates)
        self._save_last_checkpoint(trainer, monitor_candidates)
