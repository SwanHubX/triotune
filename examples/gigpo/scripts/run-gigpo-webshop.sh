#!/usr/bin/env bash
# 从任意工作目录使用 GiGPO 配置中的任务批次和 minibatch 设置启动训练。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export JAVA_HOME="${JAVA_HOME:-${REPO_ROOT}/.cache/webshop-java}"
exec "${PYTHON:-${REPO_ROOT}/.venv/bin/python}" -m examples.gigpo.train \
  --config examples/gigpo/configs/gigpo-webshop-configs.yaml \
  "$@"
