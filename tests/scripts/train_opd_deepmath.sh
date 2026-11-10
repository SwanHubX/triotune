#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec uv run python tests/recipes/train_opd.py \
  --config tests/configs/opd_deepmath.yaml "$@"
