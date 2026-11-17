#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"
EXAMPLE_ROOT="${REPO_ROOT}/examples/agent/webshop-tcod"
PROJECT_ROOT="${REPO_ROOT}"

cd "${REPO_ROOT}"
source "${EXAMPLE_ROOT}/eval/scripts/setup-java.sh" "$@"
source "${EXAMPLE_ROOT}/scripts/render-config.sh"
render_config_args "${SCRIPT_DIR}/../configs/vanilla-opd-grpo-teacher.yaml" "$@"
trap 'rm -f -- "${RENDERED_CONFIG}"' EXIT
export PYTHONPATH="${REPO_ROOT}:${EXAMPLE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON:-${REPO_ROOT}/.venv/bin/python}" \
  "${SCRIPT_DIR}/../vanilla_opd.py" \
  --config "${RENDERED_CONFIG}" \
  "${CONFIG_ARGS[@]}"
