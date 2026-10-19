#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec uv run python trio-cookbook/recipes/train_ppo.py --rebuild-data \
  --config trio-cookbook/configs/ppo_gsm8k.yaml "$@"
