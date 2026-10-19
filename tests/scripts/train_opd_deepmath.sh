#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec uv run python triotune/recipes/train_opd.py \
  --config triotune/configs/opd_deepmath.yaml "$@"
