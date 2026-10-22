#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CHECKPOINT:?Set CHECKPOINT to a trained JointHOI checkpoint}"
conda run -n jointhoi python sample.py \
  --config configs/grab.yaml \
  --checkpoint "$CHECKPOINT" \
  --prompt "Lift the mug with both hands." \
  --object mug \
  --hands both \
  --nframes 196 \
  --output outputs/grab_sample.npz \
  "$@"
