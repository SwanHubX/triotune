#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec uv run python trio-cookbook/recipes/grpo_gms8k_trio.py \
  --config trio-cookbook/configs/grpo_gms8k.toml "$@"
