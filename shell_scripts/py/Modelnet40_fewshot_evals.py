import os
import signal
import subprocess
import sys

# Environment variables
os.environ["CUDA_MATMUL_TF32"] = "high"
os.environ["LOG_LEVEL"] = "INFO"
os.environ["WARNING_LOG_FILE"] = "train.err"


# Handle Ctrl+C
def handle_sigint(sig, frame):
    print("\nInterrupted! Stopping all processes...")
    sys.exit(1)


signal.signal(signal.SIGINT, handle_sigint)

n_shots = [10, 20]
n_ways = [5, 10]
folds = range(10)

# Arguments passed to the script
additional_args = sys.argv[1:]

# Run the command for each dataset and iteration
for n_shot in n_shots:
    for n_way in n_ways:
        for fold in folds:
            print(f"Running ModelNetFewshot {n_shot}-shot, {n_way}-way, fold {fold}...")
            command = [
                "python",
                "./asymdsd/run/classification_cli.py",
                "fit",
                "--config",
                "configs/classification/classification.yaml",
                "--config",
                "configs/classification/variants/fewshot.yaml",
                "--data",
                "configs/data/ModelNet40_fewshot-S.yaml",
                "--data.init_args.dataset",
                f"data/ModelNetFewShot/ModelNetFewShot_w{n_way}_s{n_shot}_f{fold}.zarr",
                "--data.init_args.dataset_builder.init_args.n_shot",
                f"{n_shot}",
                "--data.init_args.dataset_builder.init_args.n_way",
                f"{n_way}",
                "--data.init_args.dataset_builder.init_args.fold",
                f"{fold}",
                *additional_args,
            ]

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error while running: {e}")
                sys.exit(1)
