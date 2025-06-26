#!/bin/bash

export CUDA_MATMUL_TF32='high'
export LOG_LEVEL='INFO'
export WARNING_LOG_FILE='train.err'

# Default number of runs
runs=1

# Handle Ctrl+C
trap "echo 'Interrupted! Stopping all processes...'; exit 1" SIGINT

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --runs) runs="$2"; shift 2 ;;
        *) break ;;
    esac
done

# Loop over the number of runs
for ((i=0; i<$runs; i++)); do
    echo "Running $dataset iteration $i"
    python ./asymdsd/run/classification_cli.py fit \
    --config configs/classification/classification.yaml \
    --seed_everything $i \
    "$@"
done