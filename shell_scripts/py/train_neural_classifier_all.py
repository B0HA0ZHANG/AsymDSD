import os
import signal
import subprocess
import sys

# Environment variables
os.environ["CUDA_MATMUL_TF32"] = "high"
os.environ["LOG_LEVEL"] = "INFO"
os.environ["WARNING_LOG_FILE"] = "train.err"

# Checkpoint file
checkpoint_file = "train_checkpoint.txt"


# Handle Ctrl+C
def handle_sigint(sig, frame):
    print("\nInterrupted! Stopping all processes...")
    sys.exit(1)


signal.signal(signal.SIGINT, handle_sigint)

# Parse command-line arguments
runs = 1
args = sys.argv[1:]
i = 0
additional_args = []
while i < len(args):
    if args[i] == "--runs":
        runs = int(args[i + 1])
        i += 2
    else:
        additional_args.append(args[i])
        i += 1


# Datasets to loop through
datasets = [
    "ModelNet40-S",
    "ScanObjectNN-S",
    "ScanObjectNN_OBJ_ONLY-S",
    "ScanObjectNN_OBJ_BG-S",
]

configs = {
    "configs/classification/variants/linear_from_pretrained.yaml": datasets,
    "configs/classification/variants/mlp3_from_pretrained.yaml": datasets,
    "configs/classification/variants/finetune_from_pretrained.yaml": [
        "ScanObjectNN-S",
        "ScanObjectNN_OBJ_ONLY-S",
        "ScanObjectNN_OBJ_BG-S",
    ],
    "configs/classification/variants/finetune_two_stage_from_pretrained.yaml": [
        "ModelNet40-S"
    ],
}


# Read checkpoint if it exists
def read_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return f.read().strip().split()
    return None


# Write checkpoint to file
def write_checkpoint(config, dataset, run):
    with open(checkpoint_file, "w") as f:
        f.write(f"{config} {dataset} {run}\n")


# Get the checkpoint if available
checkpoint = read_checkpoint()
if checkpoint:
    last_config, last_dataset, last_run = (
        checkpoint[0],
        checkpoint[1],
        int(checkpoint[2]),
    )
else:
    last_config, last_dataset, last_run = (
        None,
        None,
        -1,
    )  # Default values if no checkpoint exists

# Flag to indicate whether to skip or resume
resuming = last_config is not None

# Iterate over configs and datasets
for config in configs.keys():
    for dataset in configs[config]:
        if resuming:
            # Skip until we reach the checkpoint config and dataset
            if config == last_config and dataset == last_dataset:
                resuming = False  # Found the checkpoint; stop skipping
                start_run = last_run + 1  # Resume from the next run
            else:
                continue  # Skip this config/dataset combination
        else:
            start_run = 0  # Start from the first run if not resuming

        for run in range(start_run, runs):
            print(f"Running {dataset} with {config} --- run {run}...")
            command = [
                "python",
                "./asymdsd/run/classification_cli.py",
                "fit",
                "--config",
                "configs/classification/classification.yaml",
                "--data",
                f"configs/data/{dataset}.yaml",
                "--config",
                config,
                "--seed_everything",
                str(run),
                *additional_args,
            ]

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error while running: {e}")
                sys.exit(1)

            write_checkpoint(config, dataset, run)

# Remove the checkpoint file if the run was successful
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
    print("Checkpoint file removed.")
