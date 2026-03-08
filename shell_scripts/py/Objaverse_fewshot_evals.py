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

n_shots = [10]
folds = range(10)

# Arguments passed to the script
additional_args = sys.argv[1:]

# Run the command for each dataset and iteration
for n_shot in n_shots:
    for fold in folds:
        print(f"Running Objaverse LVIS few-shot, fold {fold}...")
        command = [
            "python",
            "-m",
            "asymdsd.run.classification_cli",
            "fit",
            "--config",
            "configs/classification/classification.yaml",
            "--config",
            "configs/classification/variants/objaverse_lvis.yaml",
            "--data",
            "configs/data/ObjaVerse_LVIS-S.yaml",
            "--data.init_args.supervision_key",
            f"lvis_s{n_shot}",
            "--data.init_args.split_map.train",
            f"lvis_train_s{n_shot}f{fold}",
            "--data.init_args.split_map.test",
            f"lvis_val_s{n_shot}f{fold}",
            *additional_args,
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error while running: {e}")
            sys.exit(1)
