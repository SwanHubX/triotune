# PyTRIO GRPO 训练器

本目录提供基于 PyTRIO 客户端的纯文本 GRPO 训练器和 GSM8K 示例。源码目录名为
`trio-cookbook`，Python 导入名为 `trio_cookbook`。

## 目录结构

```text
trio-cookbook/
├── configs/          # 训练参数；grpo_gms8k.toml 是 GSM8K 默认配置
├── recipes/          # 训练入口；框架版 grpo_gms8k_trio.py 和独立版 grpo_gsm8k.py
├── scripts/          # 启动脚本 train_grpo_gms8k.sh 及使用说明
├── trainer/          # GRPO 核心：输入、采样、奖励、优势、Datum 和训练循环
└── utils/            # PyTRIO 客户端及同步/异步适配工具
```

训练器模块的详细职责见 [trainer/README.md](trainer/README.md)；脚本参数及独立版示例见
[scripts/README.md](scripts/README.md)。

## 数据格式

GSM8K 训练入口读取 JSONL，每行至少包含 `messages` 和字符串 `answer`：

```json
{"messages":[{"role":"user","content":"What is 2 + 2?"}],"answer":"4"}
```

`messages` 也可以是文本或非空整数 token 列表。直接使用 `GRPOTrainer` 时，可通过
`GRPOConfig(prompt_key="prompt")` 读取旧数据的 `prompt` 字段；GSM8K 训练入口固定
校验 `messages`，旧文件需改字段名或用 `--rebuild-data` 重新生成。

## 启动命令

在仓库根目录运行（启动脚本使用 `uv`）：

```bash
# 准备并检查数据，不启动训练
bash trio-cookbook/scripts/train_grpo_gms8k.sh --prepare-only

# 按 configs/grpo_gms8k.toml 启动训练
bash trio-cookbook/scripts/train_grpo_gms8k.sh
```

启动脚本固定传入 `--rebuild-data`，每次都会重新生成并覆盖配置中的数据文件。
使用自定义数据时，请参考 [脚本说明](scripts/README.md) 直接运行训练入口。
