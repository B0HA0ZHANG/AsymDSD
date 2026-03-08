#!/usr/bin/env sh
set -eu

print_usage() {
  echo "Usage: $0 [args...]"
  echo
  echo "Runs semantic segmentation (ShapeNetPart) using configs/semseg/semseg.yaml."
  echo "You must pass: --model.encoder_ckpt_path <ckpt_path>"
  echo
  echo "Example:"
  echo "  $0 --model.encoder_ckpt_path checkpoints/AsymDSD-S_ShapeNet.ckpt"
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

python -m asymdsd.run.sem_seg_cli fit \
  --config configs/semseg/semseg.yaml \
  "$@"
