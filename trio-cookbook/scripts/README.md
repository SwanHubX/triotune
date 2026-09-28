# 训练脚本

本目录存放可直接运行的 PyTRIO 训练案例。命令默认从仓库根目录
`trio-cookbook/` 执行，并使用 `uv` 管理项目环境。

## gms8k 案例

该脚本使用 GSM8K 数据集运行 GRPO 训练。首次执行会将数据集缓存到
`.cache/huggingface/`，并生成供训练器读取的 `data/gsm8k_train.jsonl`。

### 启动训练

下面的命令运行 100 个训练 step，每个 batch 包含 4 个问题，每个问题采样
8 个 completion，并关闭 SwanLab 日志：

```bash
uv run python trio-cookbook/scripts/grpo_gms8k_trio.py \
  --steps 100 \
  --batch-size 4 \
  --group-size 8 \
  --mode disabled
```

`--mode` 控制 SwanLab 日志模式，可选 `disabled`、`online`、`local` 和
`offline`；启用后，训练结束时会以文本记录最终 checkpoint 地址，并展示最后
一批中最多 4 个问题的最终模型推理结果。训练本身仍由 PyTRIO 客户端执行。
最终步数、核心指标和 checkpoint 地址会在 SwanLab 结束前打印到终端，
同时记录为 `trainer/final_summary` 文本。需要重新下载并生成数据时，
追加 `--rebuild-data`。完整参数可以运行：

```bash
uv run python trio-cookbook/scripts/grpo_gms8k_trio.py --help
```
