from datetime import datetime
from pathlib import Path
from pickle import dump
from typing import Any, Mapping

import lightning as L
import torch


class RecordMemory(L.Callback):
    def __init__(
        self,
        num_batches: int = 3,
        log_dir: str = "logs/memory",
    ) -> None:
        self.num_batches = num_batches
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = Path(log_dir) / f"memory_usage_{now}.pickle"

        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        torch.cuda.memory._record_memory_history()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx == self.num_batches - 1:
            torch.cuda.memory._save_memory_usage(str(self.log_file))
            # torch.cuda.memory._record_memory_history(enabled=None)
            s = torch.cuda.memory._snapshot()
            with open(self.log_file, "wb") as f:
                dump(s, f)

            self.on_train_batch_end = lambda *args, **kwargs: None
