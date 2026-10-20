# OPD PyTRIO 框架与独立脚本评测

本文读取 SwanLab 0.10.1 的本地 `.swanlab` 文件，通过 `DataStoreReader` 解析 protobuf 标量、结束状态和配置，比较 DeepMath OPD 的两次实际运行。

| 运行 | 本地日志目录 | 入口 |
| --- | --- | --- |
| A：框架版 | `swanlog/run-20261010_115950-prvgeouh` | `opd_deepmath_trio.py` |
| B：独立版 | `triotune/recipes/swanlog/run-20261010_115926-y6ez99w4` | `opd_async.py` |

## 配置与完成情况

两边均使用学生 `Qwen/Qwen3.5-4B`、教师 `Qwen/Qwen3.8-27B`、LoRA rank 32、学习率 `4e-5`、Adam β₁/β₂ 为 `0.9/0.95`、seed 42、KL 信号系数 1、最大生成长度 512，以及温度 1、top-p 1、关闭 top-k；`enable_thinking=false`。框架版的 `discount_factor=0`。两边均由学生生成回答，再由固定教师对相同 token 序列评分，使用 `importance_sampling` 更新学生。

| 项目 | A：框架版 | B：独立版 |
| --- | ---: | ---: |
| 配置训练步数 | 10 | 100 |
| 实际记录的完整 step | 10（step 1–10） | 12（step 0–11） |
| 每步 prompt 数 | 2 | 4 |
| 每个 prompt 生成数 | 2 | 4 |
| 每步 completion / Datum 数 | 4 | 16 |
| prompt + completion 长度上限 | 2048 | 未记录该限制 |
| prompt 池大小 | SwanLab 配置未记录 | 5000（一个 parquet 分片抽样） |
| 结束状态 | `RUN_STATE_FINISHED` | `RUN_STATE_ABORTED` |

B 的结束记录明确为 `KeyboardInterrupt by user`，不能将其标记为完成 100 步或训练代码失败。B 未记录最终权重保存；A 记录了最终 checkpoint 路径。以下统计只覆盖已上报的完整 step，不包含中断时可能正在进行的工作。

两份元数据记录了同一台日志主机 `trio-dev02`、8 张 A800 80 GB、Python 3.12.14，以及相同的 PyTRIO 0.2.9、SwanLab 0.10.1、datasets 5.0.1 和 NumPy 2.5.3。这是客户端环境记录，不能据此确认远端训练服务的 GPU 分配或负载相同。两次运行在时间上重叠，也可能存在服务端资源竞争。

## 核心结果

均值分别覆盖 A 的 10 步、B 的 12 步；两次每步采样规模不同，不进行逐步数值复现比较。completion 吞吐统一按「已记录 completion token 总数 ÷ step 耗时合计」计算，不混用含 prompt 的输入 token 吞吐。

| 指标 | A：框架版 | B：独立版 |
| --- | ---: | ---: |
| 已记录 prompt 总数 | 20 | 48 |
| 已记录 completion 总数 | 40 | 192 |
| completion token 总数 | 20,480 | 98,304 |
| 每条 completion 平均长度 | 512 | 512 |
| 后端 `token_count` 合计 | 20,185 | 96,961 |
| 后端 `loss_mean` 的逐步均值 | 0.133821 | 0.129803 |
| sampled reverse KL 的逐步均值 | 0.132116 | 0.127956 |
| sampled reverse KL：首步 / 末步 | 0.144881 / 0.123672 | 0.134732 / 0.133771 |
| sampled reverse KL：最小 / 最大 | 0.081952 / 0.179769 | 0.099555 / 0.154867 |
| 每步耗时均值 | 12.6462 s | 12.3365 s |
| 每步耗时中位数 | 12.3267 s | 12.3149 s |
| step 耗时合计 | 126.46 s | 148.04 s |
| completion 吞吐 | 161.95 token/s | 664.04 token/s |

框架版额外记录 `time/train_runtime=127.76 s`，包含训练循环和最终状态保存等开销；独立版没有同口径的训练总耗时。表中的 step 耗时合计不能当作包含数据下载、客户端初始化和最终保存的端到端耗时。

### loss 与 KL 的含义

A 的原始字段为 `loss_mean`、`train/surrogate_loss`，B 为 `trainer/loss_mean`。A 的两个字段在全部 10 步中完全相等；当前代码已将 `train/surrogate_loss` 改名为 `train/loss`，历史记录仍使用旧名称。两边每步均满足后端定义：

```text
loss_mean = loss_sum / token_count
```

例如 A 首步 `loss_sum=295.894634`、`token_count=2023`，得到 `loss_mean=0.146265`。数百量级的是损失总和，不是平均损失。B 每步生成量是 A 的 4 倍，因此 `loss_sum` 也更大，不能直接据此判断训练效果。

A 的 `distill/reverse_kl_sampled` 与 B 的 `opd/reverse_kl_mean` 都表示采样 token 的 `log p_student − log p_teacher` 均值；A 的 `distill/advantage_mean` 在每步都等于其负值。历史运行中 B 另记录了 token 信号标准差 `opd/reverse_kl_std`，逐步均值为 0.635680；当前框架已补充同口径的 `distill/reverse_kl_sampled_std`，但该历史 A 运行尚无此字段。

`loss_mean` 是策略更新的代理损失，sampled reverse KL 是采样估计，均不是完整词表 KL。两边的后端 `token_count` 都小于生成 token 数，因此平均损失与 sampled reverse KL 也不是数值完全相同的指标。日志没有说明后端具体排除了哪些 token，不能仅凭计数差推断原因。

### 耗时与生成行为

A 每步的平均耗时分解如下，均为客户端调用的实际等待时间：

| 阶段 | 平均耗时 / 步 |
| --- | ---: |
| prompt 准备 | 0.0047 s |
| 保存学生权重并获取采样器 | 1.0396 s |
| 学生生成与教师评分 | 8.6402 s |
| 前向与反向 | 2.7167 s |
| optimizer 更新 | 0.2449 s |

学生生成与教师评分占 A 已记录 step 耗时的 68.32%，是本次主要耗时阶段。B 仅记录 step 总耗时，无法与 A 的分项等待时间对应。虽然 B 的 completion 吞吐约为 A 的 4.10 倍，但它每步生成量也是 A 的 4 倍；该结果不能独自证明独立实现具有相同 batch 下的速度优势。

两次运行的所有已记录回答长度都达到 512 token 上限。日志没有逐条 EOS 或停止原因，不能断言它们全部被截断，也不能确认回答已经完整；后续应记录结束原因和答案准确率。

## 结论与范围

两次运行的平均 loss、采样 KL 和每步耗时接近，但 batch、每题生成数及实际训练步数不同，尚不能验证框架与独立案例的一致性。后续应对齐训练配置和输入，重新比较训练指标与吞吐，再判断是否需要调整框架。
