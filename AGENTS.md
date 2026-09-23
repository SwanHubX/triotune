# AGENTS.md

本文件适用于仓库根目录及其所有子目录；子目录存在更具体的 `AGENTS.md` 时，
同时遵守子目录规则。

## 分支与范围

1. 从最新的 `main` 创建独立分支，不直接向 `main` 提交，也不对 `main` 做
   force-push。
2. 分支名使用 `<type>/<kebab-case-summary>`，例如：
   `docs/pytrio-rl-framework`、`feat/grpo-runner`、`fix/datum-alignment`。
3. `type` 优先使用 `feat`、`fix`、`docs`、`test`、`perf`、`refactor`、`ci`
   或 `chore`，并与改动性质一致。
4. 一个 PR 只解决一个明确问题。功能、重构或实验跨度过大时拆成可独立评审的
   PR，不夹带顺手修复。
5. PR 默认以 `main` 为目标分支。仅在有意构建 stacked PR 时使用其他 base，且必须
   在正文中写明依赖关系和最终合入顺序。

## Commit 规则

1. commit 标题采用 Conventional Commits：
   `<type>(<scope>): <summary>`；没有合适 scope 时可写
   `<type>: <summary>`。
2. `type` 与分支类型保持一致。常用含义如下：
   - `feat`：新增用户可见能力；
   - `fix`：修复缺陷或错误行为；
   - `docs`：仅文档改动；
   - `test`：仅测试或验收能力；
   - `perf`：有证据的性能优化；
   - `refactor`：不改变承诺行为的结构调整；
   - `ci`：工作流或构建流水线；
   - `chore`：依赖、工具或维护性工作。
3. scope 使用稳定、具体的模块名，例如 `framework`、`grpo`、`docs`、`ci`，
   不使用 `misc`、`update` 等模糊名称。
4. summary 应简短、准确并描述结果，不写 `WIP`、`fix stuff`、`update files`
   等无信息标题；同一个 PR 内的标题语言保持一致。
5. 每个 commit 表达一个可解释的逻辑单元。复杂 PR 可以包含多个 commit，但每个
   commit 都必须使用上述格式；提交前清理临时、`fixup!` 和无意义 merge commit。
6. 测试和文档可以与对应实现放在同一 commit；若单独提交，分别使用 `test(scope)`
   和 `docs(scope)`，且不能脱离相关实现形成不可用的中间状态。

示例：

```text
docs(framework): describe staged PyTRIO RL validation
feat(grpo): add group-relative advantage estimator
fix(datum): align completion logprobs with target tokens
test(checkpoint): cover interrupted resume equivalence
```

## PR 标题与正文

1. PR 标题采用与 commit 相同的 Conventional Commits 格式。本规范要求使用
   squash merge，并以 PR 标题作为 `main` 上最终 commit 的标题。
2. PR 正文至少包含：
   - **背景/目标**：为什么需要本次改动；
   - **改动内容**：实际改变了什么；
   - **验证**：执行过的命令和可核对的结果；
   - **范围/边界**：明确没有包含什么。
3. 存在兼容性、恢复、性能或外部依赖风险时，增加“风险”“兼容性”或“已知差异”
   小节，不通过放宽阈值或省略失败结果掩盖问题。
4. 性能 PR 必须同时给出测试环境、基线、优化后数据、正确性对比和资源代价；没有
   可复现实验数据时，不在标题或正文中宣称性能提升。
5. 根据 review 追加修改时，保持 commit 规则，并在正文中补充关键 review 修复及
   新增验证证据。

## 验证与提交前检查

1. 提交前检查 `git status --short`、`git diff` 和 `git diff --check`，确认文件列表、
   改动范围、空白字符和生成物符合预期。
2. 先运行与改动直接相关的最小测试，再根据风险补充更广的回归测试。文档改动至少
   检查 Markdown 格式、链接和示例命令；代码改动需覆盖成功、失败及边界路径。
3. PR 正文只记录实际执行过的验证，附命令和结果；无法运行的测试要明确写成
   “未运行”并说明原因，不把预期结果表述成已通过。
4. 修改公开契约、配置、数据结构或行为时，同步更新相关测试、示例和文档。
5. PR 创建或更新后再次检查 base/head、提交列表和 changed files，确认没有混入
   其他分支或本地工作区内容。

## 文件与产物约束

1. 不提交无关文件、本地参考仓库、临时脚本、编辑器缓存、密钥或个人环境配置。
2. 不提交模型权重、checkpoint、运行日志、实验输出、数据集缓存、构建产物或其他
   大型二进制文件；确需版本化的测试资产必须在 PR 正文中说明用途和体积。
3. 不覆盖、删除或重写他人的未提交改动。发现工作区已有改动时，将本次修改限制在
   必要文件，并在提交前核对差异。

## Review 与合入

1. `main` 上的改动必须通过 PR，并至少获得一名 reviewer 的 approval；作者不能用
   直接推送绕过评审。
2. review 指出的正确性、范围或验证问题应在合入前处理；若不采纳，需在 PR 中给出
   可核对的理由。
3. 使用 squash merge，使一个 PR 在 `main` 上对应一个最终 commit。合入前确保 PR
   标题符合 commit 规则，正文能作为最终 commit body 准确描述改动和验证证据。
