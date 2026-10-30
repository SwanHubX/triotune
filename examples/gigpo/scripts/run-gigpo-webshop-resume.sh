#!/usr/bin/env bash
# 从旧 step-50 checkpoint 续训到配置中的总 step 100。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export JAVA_HOME="${JAVA_HOME:-${REPO_ROOT}/.cache/webshop-java}"
exec "${PYTHON:-${REPO_ROOT}/.venv/bin/python}" -m examples.gigpo.train \
  --config examples/gigpo/configs/gigpo-webshop-resume-configs.yaml \
  "$@"
