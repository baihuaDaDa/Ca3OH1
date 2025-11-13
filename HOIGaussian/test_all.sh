#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Set CUDA architecture to suppress warning (adjust for your GPU)
export TORCH_CUDA_ARCH_LIST="8.6"

# Configuration
EXP_NAME=$(date +"%Y%m%d_%H%M%S")
DATA_PATH="/home/xhsystem/Code/Term7/Ca3OH1/data_sandwich"
LOG_DIR="$(pwd)/output/hoi3d_${EXP_NAME}"
LOG_FILE="$LOG_DIR/test_all.log"
mkdir -p "$LOG_DIR"

# Categories to process (keep items quoted to allow spaces)
categories=("bottle" "chain" "sandwich" "wine glass")

# Run training for one subdirectory (arguments: category, full_dir_path)
run_train() {
  local cat="$1"
  local full_path="$2"
  local dir_path
  local dir_name
  dir_path=$(dirname -- "$full_path")
  dir_name=$(basename -- "$full_path")

  echo "Data path: $dir_path"
  echo "  Training on directory: $dir_name"

  # Run training and append both stdout and stderr to log
  python3 ./train.py \
    --file_name "$dir_name" \
    --data_path "$dir_path" \
    --exp_name "hoi3d_${EXP_NAME}" \
    --motion_offset_flag \
    --smpl_type smplx \
    --iterations 160 >> "$LOG_FILE" 2>&1 || {
      echo "  Warning: training failed for $dir_name (see $LOG_FILE)" >&2
    }
}

# Main loop: iterate categories and their subdirectories safely
for cat in "${categories[@]}"; do
  full_path="$DATA_PATH/$cat"
  if [ ! -d "$full_path" ]; then
    echo "Warning: category path not found: $full_path" >&2
    continue
  fi

  for dir in "$full_path"/*; do
    [ -d "$dir" ] || continue
    run_train "$cat" "$dir"
  done
done

echo "All done. Logs appended to: $LOG_FILE"