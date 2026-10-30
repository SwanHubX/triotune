#!/usr/bin/env bash
# 从任意工作目录启动正式 GRPO WebShop 训练；额外参数原样透传给 CLI。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export JAVA_HOME="${JAVA_HOME:-${REPO_ROOT}/.cache/webshop-java}"
exec "${PYTHON:-${REPO_ROOT}/.venv/bin/python}" -m examples.gigpo.train \
  --config examples/gigpo/configs/grpo-webshop-configs.yaml \
  "$@"
