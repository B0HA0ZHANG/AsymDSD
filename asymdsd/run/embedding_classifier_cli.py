from asymdsd.components.utils import set_cuda_float32_matmul_from_env_var
from asymdsd.data import SupervisedPCDataModule
from asymdsd.models import BaseEmbeddingClassifier
from asymdsd.run.cli import TrainerCLI
from asymdsd.trainers import EmbeddingClassifierTrainer

set_cuda_float32_matmul_from_env_var()


def cli_main():
    TrainerCLI(
        model_class=BaseEmbeddingClassifier,
        datamodule_class=SupervisedPCDataModule,
        trainer_class=EmbeddingClassifierTrainer,
        subclass_mode_data=True,
        subclass_mode_model=True,
    )


if __name__ == "__main__":
    cli_main()
