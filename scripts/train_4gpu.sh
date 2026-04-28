#!/usr/bin/env bash
# 4-GPU LoRA fine-tune via DDP (one full model copy per GPU).
#
# Usage:
#   bash scripts/train_4gpu.sh                 # full run on GPUs 0-3
#   bash scripts/train_4gpu.sh --debug         # smoke test
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/train_4gpu.sh   # pick GPUs
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

torchrun --standalone --nproc_per_node=4 scripts/03_train.py --multi-gpu "$@"
