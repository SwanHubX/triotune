#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"
EXAMPLE_ROOT="${REPO_ROOT}/examples/agent/webshop-opd"

# Exercise the unified repository package with WebShop adapters.  Root test
# support supplies deterministic fake clients and a calculator environment;
# no network, model weights, WebShop data, Java, or optimizer service is used.
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${EXAMPLE_ROOT}:${REPO_ROOT}/tests${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-${REPO_ROOT}/.venv/bin/python}" -m unittest discover \
  -s "${EXAMPLE_ROOT}/tests" -p 'test_*.py' -v
