#!/usr/bin/env sh
set -eu

print_usage() {
  echo "Usage: $0 --model.encoder_ckpt_path <ckpt_path> [args...]"
  echo
  echo "Runs local ScanObjectNN fine-tuning for checkpoints pretrained with"
  echo "encoder relative_3d_bias enabled."
  echo
  echo "Base configs:"
  echo "  - configs/classification/classification.yaml"
  echo "  - configs/classification/variants/finetune_from_pretrained.yaml"
  echo "  - configs/data/ScanObjectNN-S.yaml"
  echo
  echo "Built-in encoder overrides:"
  echo "  - relative_3d_bias.class_path = asymdsd.layers.Relative3DBiasConfig"
  echo "  - num_heads = 6"
  echo "  - hidden_dim = 64"
  echo "  - use_distance = true"
  echo
  echo "Example:"
  echo "  $0 --model.encoder_ckpt_path checkpoints/Pretrain_3D_bias_only-epoch=299-step=122700.ckpt"
  echo
  echo "Useful overrides:"
  echo "  --trainer.logger.init_args.name Cls_3D_bias_only"
  echo "  --data configs/data/ScanObjectNN_OBJ_ONLY-S.yaml"
  echo "  --data configs/data/ScanObjectNN_OBJ_BG-S.yaml"
  echo "  --seed_everything 0"
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
  --model.point_encoder.init_args.encoder.relative_3d_bias.class_path asymdsd.layers.Relative3DBiasConfig \
  --model.point_encoder.init_args.encoder.relative_3d_bias.init_args.num_heads 6 \
  --model.point_encoder.init_args.encoder.relative_3d_bias.init_args.hidden_dim 64 \
  --model.point_encoder.init_args.encoder.relative_3d_bias.init_args.use_distance true \
  "$@"
