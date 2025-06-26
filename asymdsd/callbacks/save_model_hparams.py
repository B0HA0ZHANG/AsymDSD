import lightning as L
import torch
from jsonargparse import Namespace
from lightning.pytorch.cli import LightningArgumentParser


class SaveModelHparams(L.Callback):
    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        **ignore_kwargs,
    ) -> None:
        self.parser = parser
        self.config = config
        self.already_processed = False

    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        if self.already_processed:
            return

        model_hparams = self.config.model
        # Unnecessary to save this
        model_hparams.pop("__path__", None)

        if hasattr(self.config, "optim"):
            model_hparams.optim = self.config.optim
            model_hparams["optim"].pop("__path__", None)

        if hasattr(self.config, "data"):
            model_hparams.data = self.config.data
            model_hparams["data"].pop("__path__", None)
            model_hparams["data"].pop("seed", None)

        if hasattr(self.config, "trainer"):
            model_hparams.trainer = self.config.trainer
            model_hparams["trainer"].pop("__path__", None)

        model_hparams = model_hparams.as_dict()

        model_hparams["seed"] = self.config.seed_everything
        model_hparams["FP32_matmul_precision"] = torch.get_float32_matmul_precision()
        model_hparams["precision"] = trainer.precision

        # Passing Namespace will not be recursively unpacked,
        #  therefore as dict.
        pl_module.save_hyperparameters(model_hparams)

        self.already_processed = True
