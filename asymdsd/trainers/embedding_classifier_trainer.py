from datetime import timedelta
from typing import Iterable

import lightning as L
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.strategies import Strategy


class EmbeddingClassifierTrainer(L.Trainer):
    def __init__(
        self,
        *,
        accelerator: str | Accelerator = "auto",
        strategy: str | Strategy = "auto",
        devices: list[int] | str | int = "auto",
        num_nodes: int = 1,
        precision=None,
        logger: Logger | Iterable[Logger] | bool | None = None,
        callbacks: list[Callback] | Callback | None = None,
        fast_dev_run: int | bool = False,
        max_time: str | timedelta | dict[str, int] | None = None,
        limit_train_batches: int | float | None = None,
        limit_val_batches: int | float | None = None,
        limit_test_batches: int | float | None = None,
        limit_predict_batches: int | float | None = None,
        overfit_batches: int | float = 0,
        log_every_n_steps: int | None = None,
        enable_progress_bar: bool | None = None,
        enable_model_summary: bool | None = None,
        deterministic=None,
        benchmark: bool | None = None,
        use_distributed_sampler: bool = True,
        profiler: Profiler | str | None = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins=None,
        default_root_dir: _PATH | None = None,
    ) -> None:
        max_epochs = 1
        enable_checkpointing: bool | None = False
        num_sanity_val_steps: int | None = 0

        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            logger=logger,
            callbacks=callbacks,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            max_time=max_time,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=overfit_batches,
            num_sanity_val_steps=num_sanity_val_steps,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            deterministic=deterministic,
            benchmark=benchmark,
            use_distributed_sampler=use_distributed_sampler,
            profiler=profiler,
            detect_anomaly=detect_anomaly,
            barebones=barebones,
            plugins=plugins,
            default_root_dir=default_root_dir,
        )
