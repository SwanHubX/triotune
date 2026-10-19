#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec uv run --extra dpo python trio-cookbook/recipes/train_dpo.py \
  --config trio-cookbook/configs/dpo_rlhf.yaml "$@"
