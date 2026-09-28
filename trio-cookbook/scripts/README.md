# 训练脚本

本目录存放可直接运行的 PyTRIO 训练案例。命令默认从仓库根目录
`trio-cookbook/` 执行，并使用 `uv` 管理项目环境。

## 输入数据集格式

`grpo_gms8k_trio.py` 的 `--data-file` 接收 UTF-8 编码的 JSONL 文件：每个非空行是
一条 JSON 对象，文件至少包含一条有效记录。例如：

```jsonl
{"prompt":[{"role":"user","content":"What is 6 times 7? Answer with \\boxed{}."}],"answer":"42","question":"What is 6 times 7?"}
```

- `prompt`（必填）：非空列表。推荐使用对话消息，每条消息含字符串类型的 `role`
  和 `content`；也可以使用非空的整数 token ID 列表。此脚本的读取校验不接受
  裸字符串 `prompt`。
- `answer`（必填）：字符串形式的参考数值答案，例如 `"42"`。奖励函数提取生成文本
  中最后一个 `\boxed{}` 的内容，去掉千位逗号、首尾空白和末尾句点后比较；提示词
  应要求模型用 `\boxed{}` 作答。
- `question`（可选）：原始问题文本，用作 SwanLab 最终推理样例的标题；省略时使用
  渲染后的 `prompt`。其他字段可保留在数据行中。

已存在的 `--data-file` 会直接读取；文件不存在或指定 `--rebuild-data` 时，脚本会
下载 `openai/gsm8k` 并在该路径生成数据，因此不要对自定义文件使用
`--rebuild-data`。可先检查读取的记录数：

```bash
uv run python trio-cookbook/scripts/grpo_gms8k_trio.py \
  --data-file path/to/train.jsonl --prepare-only
```

`--prepare-only` 检查外层字段类型；对话消息或 token 列表的具体内容在训练时进一步
校验。

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
