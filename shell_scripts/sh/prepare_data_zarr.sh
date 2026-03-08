#!/usr/bin/env sh
set -eu

print_usage() {
  echo "Usage: $0 <config.yaml|dataset_name> [more configs...]"
  echo
  echo "Examples:"
  echo "  $0 configs/data/prepare_data_zarr/Objaverse.yaml"
  echo "  $0 Objaverse"
  echo "  $0 ScannedObjects Toys4K"
}

if [ "$#" -lt 1 ]; then
  print_usage
  exit 1
fi

case "${1:-}" in
  -h|--help)
    print_usage
    exit 0
    ;;
esac

for arg in "$@"; do
  cfg_path="$arg"

  if [ -f "$cfg_path" ]; then
    :
  elif [ -f "configs/data/prepare_data_zarr/$cfg_path" ]; then
    cfg_path="configs/data/prepare_data_zarr/$cfg_path"
  elif [ -f "configs/data/prepare_data_zarr/$cfg_path.yaml" ]; then
    cfg_path="configs/data/prepare_data_zarr/$cfg_path.yaml"
  else
    echo "Error: could not find config '$arg'" 1>&2
    echo "Looked for:" 1>&2
    echo "  - $arg" 1>&2
    echo "  - configs/data/prepare_data_zarr/$arg" 1>&2
    echo "  - configs/data/prepare_data_zarr/$arg.yaml" 1>&2
    exit 1
  fi

  echo "==> Preparing zarr dataset using: $cfg_path"
  python -m asymdsd.run.prepare_zarr_ds_cli --config "$cfg_path"
done
