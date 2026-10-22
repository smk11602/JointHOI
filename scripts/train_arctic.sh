#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
conda run -n jointhoi python train.py --config configs/arctic.yaml "$@"
