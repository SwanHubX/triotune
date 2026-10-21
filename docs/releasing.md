# 发布 triotune 到 PyPI

## 首次配置

1. 将 `.github/workflows/` 合并到 `main`，GitHub 才能接收分支创建事件和
   `workflow_dispatch`。
2. 在仓库 Settings → Actions → General 中允许 GitHub Actions 创建 Pull Request。
   组织策略也需要允许此操作。
3. 保护 `main`：要求通过 PR 合并、至少一名 reviewer approval，并启用 squash merge。
   使用默认 `GITHUB_TOKEN` 创建的 PR 不会触发普通 `pull_request` workflow；
   发布 PR 的测试由 `release-pr.yml` 在创建 PR 前显式运行。
4. 创建 GitHub environment `pypi`。如限制部署分支，应允许 `main`。
5. 在 PyPI 的 **triotune** 项目中添加 GitHub Trusted Publisher：

   | 字段 | 值 |
   | --- | --- |
   | Owner | `SwanHubX` |
   | Repository | `triotune` |
   | Workflow filename | `release.yml` |
   | Environment | `pypi` |

   无需保存长期 PyPI token；发布 job 使用 OIDC。若项目尚不存在，先配置同名
   pending publisher。仓库迁移后应同步调整上述配置。

## 发布步骤

确保工作区干净，从最新 `main` 创建发布分支；版本必须为严格递增的稳定版
`X.Y.Z`（无前导零），并与仓库的 `chore/<kebab-case-summary>` 分支规则一致。
例如，当前版本为 `0.1.2` 时：

```bash
git fetch origin main
git switch -c chore/release-v0.1.3 origin/main
git push -u origin chore/release-v0.1.3
```

后续流程自动执行：

1. `prepare-release.yml` 确认分支指向当前 `main`、版本递增且 tag 尚不存在；
   使用 `uv version <版本> --no-sync` 同步更新 `pyproject.toml` 和 `uv.lock`，
   仅向发布分支提交版本变更。
2. `release-pr.yml` 验证该提交的完整 SHA，执行本地单元测试、构建 sdist/wheel、
   检查包元数据及安装后的 `triotune` 版本，再创建目标为 `main` 的发布 PR。
3. 至少获得一名 reviewer approval 后，以 squash merge 合并 PR。
4. `release.yml` 对合并提交重新测试并构建；在 `pypi` environment 下创建
   `vX.Y.Z` tag、发布到 PyPI，成功后创建 GitHub Release 并上传同一份产物，
   最后清理发布分支。手工推送 tag 不会触发发布。

发布 PR 创建后请勿追加提交；需要修改时，先完成修改，再对最新分支 HEAD
重新运行 `release-pr.yml`，输入对应的 `branch`、`version` 和完整 `expected_sha`。
该流程会重新验证、复用已有 PR 并更新描述中的验证记录。

## 本地验证

CI 使用 Python 3.12，运行时依赖从 `uv.lock` 安装。DPO/PPO 的 loss 测试需要
PyTorch；CI 单独安装 CPU wheel，避免下载 CUDA 运行库。以下命令会同步当前
`.venv`。

```bash
uv sync --locked --python 3.12
uv pip install 'torch>=2.3' --index-url https://download.pytorch.org/whl/cpu
uv run --no-sync python -m unittest discover -s tests -p 'test_*.py'
uv build
uv tool run twine check dist/*
```

只收集 `test_*.py`，避免执行 `tests/trio_test.py` 和 `tests/recipes/` 中需要真实
云端凭据、数据下载或远端训练服务的手工脚本。`--no-sync` 确保执行测试时不会
移除单独安装的 CPU PyTorch。普通 PR 也使用同一套单元测试；本项目尚未配置
Ruff/basedpyright，workflow 不沿用来源项目的静态检查门禁。

发布失败时，从 GitHub Actions 重跑失败的 job。已有 tag 必须指向同一合并提交；
PyPI 上传跳过已有文件，GitHub Release 复用已有页面，全部成功后才清理分支。
`pyproject.toml` 是版本来源，`uv.lock` 中的本地项目版本必须与其一致。
