# SFT PyTRIO 框架与独立脚本评测

本文比较两次甄嬛对话 SFT 运行的 SwanLab 0.10.1 本地记录。使用 `DataStoreReader` 读取 `.swanlab` 中的标量，按相同的 step 1–100 对齐；两次运行的结束状态均为 `RUN_STATE_FINISHED`。

| 运行 | SwanLab 目录 | 元数据记录的入口与运行名 |
| --- | --- | --- |
| A（标记为框架版） | `/data/home/wzc/triotune/swanlog/run-20261009_111858-mi6d3u69` | `sft_huanhuan.py`；`huanhuan-framework-100step-seed42` |
| B（标记为独立版） | `/data/home/wzc/triotune/swanlog/run-20261009_111620-4bm2i0f3` | `sft_huanhuan_direct.py`；`huanhuan-direct-100step-seed42` |

两次运行的元数据记录了同一台主机、Qwen/Qwen3.5-4B、LoRA rank 8、batch size 1、最大长度 384、学习率 1e-4、seed 42，以及命令行中的 `--limit 100 --epochs 1 --steps 100 --save-every 0`。两份配置都将聊天模板的 `enable_thinking` 设为 `false`。以下比较只使用实际记录的指标，不将相同配置视为逐 token 输入完全一致的证明。

## 核心结果

`train/loss`、`time/step` 的均值和中位数均覆盖 100 个 step；总受训 token 数为 `train/tokens` 的逐步求和。吞吐量由总受训 token 数除以 `time/step` 之和计算，因此只表示已记录训练阶段的受训 token 吞吐。

| 指标 | A：框架版 | B：独立版 |
| --- | ---: | ---: |
| 完成 step 数 | 100 | 100 |
| 受训 token 总数 | 1,745 | 1,745 |
| `train/loss` 均值 | 3.047878 | 3.047564 |
| `train/loss`：首步 / 末步 | 6.944216 / 2.434850 | 6.944216 / 2.432263 |
| `time/step` 均值 | 1.2545 s | 1.2795 s |
| `time/step` 中位数 | 1.1769 s | 1.1941 s |
| `time/step` 合计 | 125.45 s | 127.95 s |
| 受训 token 吞吐 | 13.91 token/s | 13.64 token/s |

两边的 `train/tokens` 在 100 个对应 step 上全部相同，学习率也均为 1e-4。平均 `train/loss` 相差 0.000314，约为 B 的 0.010%；逐步绝对差的均值为 0.009994。首步 loss 完全相同，但只有 4/100 个 step 的 loss 数值完全相同；最大逐步差发生在 step 64，A 为 2.520781、B 为 2.472527，相差 0.048254（约为 B 的 1.95%）。这表明整体训练 loss 接近，并未逐步数值复现。相同的受训 token 数只能说明每步参与 loss 的标签数量相同，不能单独证明 token ID 或初始模型状态相同。

A 的 `time/step` 合计比 B 少 2.50 秒，约为 B 合计的 1.95%。去掉较慢的首步后，两边其余 99 步的平均耗时分别为 1.2333 秒和 1.2563 秒，方向相同；但差异并非每步稳定存在。例如 step 74 为 A 1.158 秒、B 2.313 秒，而 step 99 为 A 2.007 秒、B 1.173 秒。日志没有记录前反向、优化器、请求等待等分项耗时，无法据此定位这些波动。`time/step` 合计也不是包含客户端初始化和最终保存的端到端运行时间。

## 结论与范围

没啥问题，基本重合

两次运行的受训 token 数逐步一致，平均训练 loss 非常接近；但它们并未逐步数值复现，且没有验证集指标，不能据此判断泛化质量。两份元数据都指向提交 `39b709f29364f331209c9e1b7a58bdab789ace6b`，该提交未包含运行命令所指向的两个 recipe 文件；“框架版/独立版”的对应关系来自运行名和命令行，实际运行源码及差异原因无法仅凭这些日志核实。

## 修改了trainer布局之后的测试

没啥问题
