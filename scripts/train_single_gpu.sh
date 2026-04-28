#!/usr/bin/env bash
# Single-GPU LoRA fine-tune.
#
# Usage:
#   bash scripts/train_single_gpu.sh                 # full run
#   bash scripts/train_single_gpu.sh --debug         # 100-sample smoke test
#   bash scripts/train_single_gpu.sh --batch-size 1  # any extra args pass through
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python scripts/03_train.py "$@"
