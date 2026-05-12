import os
import pickle
from functools import wraps
from pathlib import Path
from typing import Sequence

import torch
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from asymdsd.callbacks import SaveModelHparams
from asymdsd.components.optimizer_spec import OptimizerSpec
from asymdsd.components.utils import compile_model as compile_model_fn
from asymdsd.defaults import DEFAULT_OPTIMIZER
from asymdsd.loggers import setup_logger


def register_checkpoint_safe_globals() -> None:
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    from asymdsd.components.encoder_branch import EncoderBranch
    from asymdsd.data.data_module import DatasetSplit
    from asymdsd.data.data_module_zarr import SubsampleMode
    from asymdsd.data.dataset_builder import FieldType, PCFieldKey
    from asymdsd.layers.classification_head import (
        ClassificationHeadType as LayerClassificationHeadType,
    )
    from asymdsd.layers.masked_center_predictor import MaskedCenterPredictorConfig
    from asymdsd.layers.semantic_slots import SemanticSlotConfig
    from asymdsd.models.asymdsd import ClsPredictor, TraingingMode
    from asymdsd.models.neural_classifier import (
        ClassificationHeadType as NeuralClassificationHeadType,
    )
    from asymdsd.models.semantic_segmentation import (
        ClassificationHeadType as SemanticClassificationHeadType,
    )

    add_safe_globals(
        [
            EncoderBranch,
            DatasetSplit,
            SubsampleMode,
            FieldType,
            PCFieldKey,
            LayerClassificationHeadType,
            MaskedCenterPredictorConfig,
            SemanticSlotConfig,
            NeuralClassificationHeadType,
            SemanticClassificationHeadType,
            TraingingMode,
            ClsPredictor,
        ]
    )


def patch_lightning_checkpoint_io() -> None:
    current = TorchCheckpointIO.load_checkpoint
    if getattr(current, "__asymdsd_weights_only_fallback__", False):
        return

    def load_checkpoint_with_fallback(
        self,
        path: str | Path,
        map_location=lambda storage, loc: storage,
    ):
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        with fs.open(path, "rb") as f:
            try:
                return torch.load(f, map_location=map_location)
            except (pickle.UnpicklingError, RuntimeError):
                f.seek(0)
                return torch.load(f, map_location=map_location, weights_only=False)

    load_checkpoint_with_fallback.__asymdsd_weights_only_fallback__ = True
    TorchCheckpointIO.load_checkpoint = load_checkpoint_with_fallback


def compile_model(func):
    @wraps(func)
    def wrapper(self, model, **kwargs):
        subcommand = self.config.subcommand
        config_kwargs = self.config.as_dict()
        compile_kwargs = config_kwargs[subcommand]["compile"]
        compile_kwargs.pop("__path__", None)
        model = compile_model_fn(model, **compile_kwargs)

        return func(self, model, **kwargs)

    return wrapper


class TrainerCLI(LightningCLI):
    def __init__(
        self,
        linked_args_list: Sequence[tuple[str | tuple[str], str]] | None = None,
        default_optimizer: OptimizerSpec = DEFAULT_OPTIMIZER,
        add_optim_key: bool = False,
        **kwargs,
    ) -> None:
        if "save_config_callback" not in kwargs:
            kwargs["save_config_callback"] = SaveModelHparams
        self.linked_args_list = linked_args_list or []
        self.default_optimizer = default_optimizer
        self.add_optim_key = add_optim_key
        register_checkpoint_safe_globals()
        patch_lightning_checkpoint_io()
        self._setup_logger()
        super().__init__(**kwargs)

    def _setup_logger(self) -> None:
        info = warning = level = None
        if "LOG_LEVEL" in os.environ:
            level = os.environ["LOG_LEVEL"]
        if "INFO_LOG_FILE" in os.environ:
            info = os.environ["INFO_LOG_FILE"]
        if "WARNING_LOG_FILE" in os.environ:
            warning = os.environ["WARNING_LOG_FILE"]

        setup_logger(level=level, info_output=info, warn_output=warning)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        for linked_args in self.linked_args_list:
            parser.link_arguments(*linked_args, apply_on="parse")

        if self.add_optim_key:
            parser.add_subclass_arguments(
                OptimizerSpec,
                "optim",
                default=self.default_optimizer,
                instantiate=True,
            )
            parser.link_arguments("optim", "model.optimizer", apply_on="instantiate")

        parser.add_function_arguments(
            compile_model_fn, skip=set(["model"]), nested_key="compile"
        )

    @compile_model
    def fit(self, model, **kwargs):
        self.trainer.fit(model, **kwargs)
