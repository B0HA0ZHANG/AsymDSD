#!/bin/bash

export CUDA_MATMUL_TF32='high'
export LOG_LEVEL='INFO'
export WARNING_LOG_FILE='train.err'

# Default to a single visible GPU. This avoids NCCL startup failures on
# machines where multi-GPU distributed training is not configured cleanly.
# Override this explicitly, e.g. CUDA_VISIBLE_DEVICES=0,1 sh ...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python ./asymdsd/run/ssrl_cli.py fit --config configs/ssrl/ssrl.yaml "$@"
