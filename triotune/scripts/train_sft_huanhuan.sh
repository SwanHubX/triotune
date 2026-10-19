#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec uv run python triotune/recipes/train_sft.py \
  --config triotune/configs/sft_huanhuan.yaml "$@"
