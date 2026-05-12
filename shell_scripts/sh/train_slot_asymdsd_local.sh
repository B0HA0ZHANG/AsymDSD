#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MATMUL_TF32="${CUDA_MATMUL_TF32:-high}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export WARNING_LOG_FILE="${WARNING_LOG_FILE:-logs/slot_asymdsd/train.err}"

mkdir -p logs/slot_asymdsd checkpoints/slot_asymdsd

PYTHON_BIN="${PYTHON_BIN:-/home/bohaozhang/miniconda3/envs/asymdsd/bin/python}"
PRETRAIN_NAME="${PRETRAIN_NAME:-Pretrain_slot_asymdsd}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-409}"
CKPT_PATH="${CKPT_PATH:-}"

args=(
  ./asymdsd/run/ssrl_cli.py
  fit
  --config configs/ssrl/ssrl.yaml
  --trainer configs/local_seq/ssrl_trainer_pretrain_only_wandb.yaml
  --model configs/ssrl/variants/model/ssrl_model_slot_asymdsd.yaml
  --data configs/data/ShapeNetCore-U.yaml
  --data.init_args.batch_size "$BATCH_SIZE"
  --trainer.accumulate_grad_batches "$ACCUMULATE_GRAD_BATCHES"
  --model.steps_per_epoch "$STEPS_PER_EPOCH"
  --trainer.logger.init_args.name "$PRETRAIN_NAME"
)

if [[ -n "$CKPT_PATH" ]]; then
  args+=(--ckpt_path "$CKPT_PATH")
fi

exec "$PYTHON_BIN" "${args[@]}" "$@"
