# Triotune SWE 自动修复流程

离线 SWE SFT 管线继续保留，在线执行由两个独立项目提供：

| 层 | 目录 / 模块 | 职责 |
| --- | --- | --- |
| 框架与模型 | `triotune/swe` | PyTRIO 采样、真实 token/logprob、策略版本、PPO 契约转换 |
| Agent 执行 | `playground/Triotune-SWEAgent` / `tswe` | 复现、模型交互、工具、失败反馈、状态与产物 |
| 环境 | `playground/Triotune-Environment` / `tenv` | Git 快照、Docker、持久 shell、编辑、patch、独立测试 |

两个 playground 目录是独立 Git 仓库，继续由主仓库原有忽略规则排除。交付时需一并取得两个仓库以及兼容的 triotune 源码。

## 安装与运行

从 triotune 源码根目录执行：

```bash
python -m pip install -e . -e playground/Triotune-Environment -e playground/Triotune-SWEAgent
docker build -t triotune-swe:py312 playground/Triotune-Environment
python playground/Triotune-SWEAgent/examples/demo.py --model Qwen/Qwen3.5-4B
```

PyTRIO 使用已有身份与服务配置；凭证不传入沙箱。沙箱默认无网络，项目依赖需预装到任务镜像内。真实任务修改执行项目的 `examples/task.yaml`，再运行：

```bash
python -m tswe playground/Triotune-SWEAgent/examples/task.yaml --output output/swe
```

## 自动修复顺序

1. 为指定 commit 创建原始快照和工作副本，保留源仓库。
2. 独立运行任务提供的测试，确认基线失败；已通过时停止。
3. 给模型发送 issue、失败观察和工具定义。
4. 执行 `execute_bash` 或 `search_replace`，回填结果，继续请求模型。
5. `task_done` 提取 patch，拒绝修改测试等受保护文件。
6. 在新工作区应用 patch、重跑原始测试；失败继续反馈，通过返回奖励 1。
7. 保存结果、轨迹与 patch，释放本次容器和临时目录。

交付产物为 `patch.diff`、`episode.json` 和可用于离线 SFT 的 `trajectory.jsonl`，不自动将 patch 应用到源仓库。失败或预算耗尽也会记录状态。

## 训练接口

`triotune.swe.PyTRIOSWEModel` 复用 `call_async` 和 PyTRIO sampling API，支持基础模型、保存的模型路径及训练客户端导出的采样快照。

`EpisodeResult.sft_record()` 与 `prepare_swe_sft_dataset()` 的 messages/tools 输入一致。新增 `SFTTrainer.get_tokenizer()` 提供公开预处理接口；预处理与 trainer 的长度限制必须统一，真实模板必须支持准确的 assistant mask。

`EpisodeResult.to_trajectory(group_id=...)` 使用已有 `Trajectory` 和 `PPOTrainer.step()`，检查有效任务状态、PPO 采样参数、单一策略版本和真实 token 前缀。历史被模板改写时明确拒绝整轨迹导出，不用重新分词代替采样记录。

本功能不改变现有 SFT/PPO 默认行为，训练本身不会自动启动 agent。详细配置与训练示例见执行项目 README；环境后端和保护规则见环境项目 README。

## 本次验证记录

以下命令已在当前工作区实际执行：

```bash
.venv/bin/python -m unittest discover -s tests -p test_swe_runtime.py -v
.venv/bin/python -m unittest discover -s tests -p test_sft_trainer.py -v
.venv/bin/python -m unittest discover -s tests -p test_swe_sft_dataset.py -v

TENV_DOCKER_IMAGE=swebench-sandbox-local:latest \
  PYTHONPATH=playground/Triotune-Environment/src \
  .venv/bin/python -m unittest discover -s playground/Triotune-Environment/tests -v

TENV_DOCKER_IMAGE=swebench-sandbox-local:latest \
  PYTHONPATH=playground/Triotune-Environment/src:playground/Triotune-SWEAgent/src \
  .venv/bin/python -m unittest discover -s playground/Triotune-SWEAgent/tests -v

PYTHONPATH=playground/Triotune-Environment/src:playground/Triotune-SWEAgent/src \
  .venv/bin/python playground/Triotune-SWEAgent/examples/demo.py \
  --model Qwen/Qwen3.5-4B --image swebench-sandbox-local:latest --output output/swe-live
```

结果：框架采样/PPO 桥接 7 项、SFTTrainer 19 项、已有 SWE SFT 6 项、环境 7 项、agent 11 项，共 50 项通过，包括 Docker 集成测试。

真实 PyTRIO `Qwen/Qwen3.5-4B` 的演示在 4 轮完成查看代码、修改、运行测试和提交；基线退出码 1，独立评测退出码 0，最终 `resolved`、`reward=1`。这是小型算术回归的端到端验证，不代表大型 SWE-bench 的修复率。

未执行真实远程 SFT 或 PPO 优化；训练桥接通过测试客户端验证，真实 tokenizer 的 assistant mask 仍取决于所选模板。
