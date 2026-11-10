#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"
EXAMPLE_ROOT="${REPO_ROOT}/examples/agent/webshop-opd"

cd "${REPO_ROOT}"
source "${EXAMPLE_ROOT}/eval/scripts/setup-java.sh" "$@"
export PYTHONPATH="${REPO_ROOT}:${EXAMPLE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-${REPO_ROOT}/.venv/bin/python}" \
  "${SCRIPT_DIR}/../collect_tcod_b2f_experts.py" \
  --config "${SCRIPT_DIR}/../configs/tcod-b2f-qwen3.8-27b-teacher.yaml" \
  "$@"
