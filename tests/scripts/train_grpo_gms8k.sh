#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec uv run python triotune/recipes/train_grpo.py --rebuild-data \
  --config triotune/configs/grpo_gms8k.yaml "$@"
