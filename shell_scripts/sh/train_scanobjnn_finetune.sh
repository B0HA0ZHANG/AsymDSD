#!/usr/bin/env sh
set -eu

print_usage() {
  echo "Usage: $0 --model.encoder_ckpt_path <ckpt_path> [args...]"
  echo
  echo "Runs local ScanObjectNN fine-tuning using:"
  echo "  - configs/classification/classification.yaml"
  echo "  - configs/classification/variants/finetune_from_pretrained.yaml"
  echo "  - configs/data/ScanObjectNN-S.yaml"
  echo
  echo "Example:"
  echo "  $0 --model.encoder_ckpt_path checkpoints/AsymDSD-S_ShapeNet.ckpt"
  echo
  echo "Useful overrides:"
  echo "  --trainer.logger.init_args.name Finetune_ScanObjectNN"
  echo "  --data configs/data/ScanObjectNN_OBJ_ONLY-S.yaml"
  echo "  --data configs/data/ScanObjectNN_OBJ_BG-S.yaml"
  echo "  --model configs/classification/variants/model/classification_model_base.yaml"
}

case "${1:-}" in
  -h|--help)
    print_usage
    exit 0
    ;;
esac

export CUDA_MATMUL_TF32='high'
export LOG_LEVEL='INFO'
export WARNING_LOG_FILE='train.err'

# Default to a single visible GPU. Override explicitly if needed, e.g.
# CUDA_VISIBLE_DEVICES=0,1 sh ...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python ./asymdsd/run/classification_cli.py fit \
  --config configs/classification/classification.yaml \
  --config configs/classification/variants/finetune_from_pretrained.yaml \
  --data configs/data/ScanObjectNN-S.yaml \
  "$@"
