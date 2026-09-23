# AGENTS.md

本规则适用于仓库根目录及所有子目录；子目录中的更具体 `AGENTS.md` 优先。仓库维护者：`828Tina`。

## 分支与改动范围

- 从最新 `main` 创建独立分支，命名为 `<type>/<kebab-case-summary>`。
- `type` 使用 `feat`、`fix`、`docs`、`refactor`、`test`、`perf`、`chore` 或 `ci`，并与改动性质一致。
- 不直接提交或 force-push 到 `main`；一个 PR 只解决一个明确问题，默认以 `main` 为目标分支。使用 stacked PR 时说明依赖关系和合并顺序。
- 不覆盖或删除他人的未提交改动。

## 提交与 Pull Request

- 所有提交和 PR 标题遵循 Angular Commit 规范：
  `<type>(<scope>): <imperative summary>`；`scope` 可选，标题使用简洁明确的英文且不超过 72 个字符。
- 提交正文（如有）和 PR 描述使用中文。每个 commit 应是一个完整、可解释的逻辑单元。
- PR 默认使用 squash merge，并至少获得一名 reviewer approval；不得通过直接推送绕过评审。
- PR 描述简明说明背景/目标、改动、验证结果和范围边界；有风险时补充风险或兼容性说明。
- 只记录实际执行的验证命令和结果；review 修复应同步更新说明。

## 验证与文件约束

- 提交前检查 `git status --short`、`git diff`、`git diff --check`，并运行与改动直接相关的最小测试。
- 文档改动检查 Markdown、链接和示例；代码改动覆盖成功、失败和边界路径。
- 修改公开契约、配置或行为时，同步更新测试、示例和文档；性能 PR 必须给出环境、基线、优化后数据、正确性对比和资源代价。
- 不提交无关文件、密钥、个人配置、缓存、日志、模型权重、checkpoint、数据集缓存、构建产物或大型二进制文件。
- PR 创建或更新后，确认 base/head、提交列表和 changed files 正确。
