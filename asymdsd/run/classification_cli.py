from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.cli import LightningArgumentParser

from asymdsd.components.utils import set_cuda_float32_matmul_from_env_var
from asymdsd.data import SupervisedPCDataModule
from asymdsd.defaults import DEFAULT_CLASSIFIER_OPTIMIZER
from asymdsd.models import NeuralClassifier
from asymdsd.run.cli import TrainerCLI

set_cuda_float32_matmul_from_env_var()

LINKED_ARGS = [
    ("data.init_args.batch_size", "model.batch_size"),
    (
        "model.point_encoder.init_args.encoder.embed_dim",
        "model.point_encoder.init_args.patch_embedding.init_args.position_embedding.init_args.embed_dim",
    ),
    (
        "model.point_encoder.init_args.encoder.embed_dim",
        "model.point_encoder.init_args.patch_embedding.init_args.point_embedding.init_args.embed_dim",
    ),
    ("trainer.max_epochs", "model.max_epochs"),
    ("trainer.max_steps", "model.max_steps"),
]

TRAINER_DEFAULTS = {
    "callbacks": LearningRateMonitor(logging_interval="step", log_weight_decay=True),
}


class ClassifierTrainerCLI(TrainerCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)


def cli_main():
    ClassifierTrainerCLI(
        model_class=NeuralClassifier,
        datamodule_class=SupervisedPCDataModule,
        trainer_defaults=TRAINER_DEFAULTS,
        linked_args_list=LINKED_ARGS,
        default_optimizer=DEFAULT_CLASSIFIER_OPTIMIZER,
        add_optim_key=True,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    cli_main()
