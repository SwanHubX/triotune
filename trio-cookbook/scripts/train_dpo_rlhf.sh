#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec uv run --extra dpo python trio-cookbook/recipes/dpo_rlhf_trio.py \
  --config trio-cookbook/configs/dpo_rlhf.toml "$@"
