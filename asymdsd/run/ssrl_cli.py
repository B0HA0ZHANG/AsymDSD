from lightning.pytorch.callbacks import LearningRateMonitor

from asymdsd import (
    AsymDSD,
    PointCloudDataModule,
)
from asymdsd.components.utils import set_cuda_float32_matmul_from_env_var
from asymdsd.models.asymdsd import TraingingMode
from asymdsd.run.cli import TrainerCLI

set_cuda_float32_matmul_from_env_var()


def get_multi_crop_config(multi_crop_config, training_mode):
    if training_mode == TraingingMode.MASK:
        return None  # Disable multi-crop for mask training
    return multi_crop_config


LINKED_ARGS = [
    ("data.init_args.batch_size", "model.batch_size"),
    # ('data.num_point_features', 'model.num_point_features'), # Legacy (might want to readd in some way)
    ("model.encoder_config.embed_dim", "model.projection_head_config.in_dim"),
    (
        "model.encoder_config.embed_dim",
        "model.patch_embedding.init_args.position_embedding.init_args.embed_dim",
    ),
    (
        "model.encoder_config.embed_dim",
        "model.patch_embedding.init_args.point_embedding.init_args.embed_dim",
    ),
    ("trainer.max_epochs", "model.max_epochs"),
    ("trainer.max_steps", "model.max_steps"),
    (
        ("data.init_args.multi_crop_config", "model.training_mode"),
        "data.init_args.multi_crop_config",
        get_multi_crop_config,
    ),
]

TRAINER_DEFAULTS = {
    "callbacks": LearningRateMonitor(logging_interval="step", log_weight_decay=True),
}


def cli_main():
    TrainerCLI(
        model_class=AsymDSD,
        datamodule_class=PointCloudDataModule,
        linked_args_list=LINKED_ARGS,
        trainer_defaults=TRAINER_DEFAULTS,
        add_optim_key=True,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    cli_main()
