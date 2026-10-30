#!/usr/bin/env bash
# Start training with the GiGPO batch and minibatch settings from any directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export JAVA_HOME="${JAVA_HOME:-${REPO_ROOT}/.cache/webshop-java}"
exec "${PYTHON:-${REPO_ROOT}/.venv/bin/python}" -m examples.gigpo.train \
  --config examples/gigpo/configs/gigpo-webshop-configs.yaml \
  "$@"
