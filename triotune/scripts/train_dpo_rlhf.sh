#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec uv run --extra dpo python triotune/recipes/train_dpo.py \
  --config triotune/configs/dpo_rlhf.yaml "$@"
