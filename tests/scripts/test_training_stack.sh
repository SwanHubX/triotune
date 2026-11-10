#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
AUDIT_PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
AUDIT_OUTPUT="${TRIOTUNE_AUDIT_OUTPUT:-$(mktemp -d -t triotune-training-audit-XXXXXXXX)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/tests${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1

# trio_test.py is an executable cloud service probe, not a regression test.
"${AUDIT_PYTHON}" -m pytest -q tests tests/audit_agentrl_functionality.py \
  --ignore=tests/trio_test.py -rs
if [[ -f examples/agent/webshop-opd/train/scripts/test-agent-opd.sh ]]; then
  PYTHON="${AUDIT_PYTHON}" bash examples/agent/webshop-opd/train/scripts/test-agent-opd.sh
fi
"${AUDIT_PYTHON}" tests/verify_agentrl_wheel.py --output "${AUDIT_OUTPUT}/dist"
