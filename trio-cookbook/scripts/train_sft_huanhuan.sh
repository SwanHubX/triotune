#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec uv run python trio-cookbook/recipes/sft_huanhuan_trio.py \
  --config trio-cookbook/configs/sft_huanhuan.toml "$@"
