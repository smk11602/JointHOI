#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CHECKPOINT:?Set CHECKPOINT to a trained JointHOI checkpoint}"
conda run -n jointhoi python sample.py \
  --config configs/arctic.yaml \
  --checkpoint "$CHECKPOINT" \
  --prompt "Open the box with both hands." \
  --object box \
  --hands both \
  --nframes 64 \
  --output outputs/arctic_sample.npz \
  "$@"
