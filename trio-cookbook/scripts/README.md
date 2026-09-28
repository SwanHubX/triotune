# 训练脚本

本目录存放训练启动脚本；训练案例实现位于
[grpo_gms8k_trio.py](../recipes/grpo_gms8k_trio.py)。
命令默认从仓库根目录执行，并使用 `uv` 管理项目环境。启动脚本会切换到仓库根目录，
再读取 `trio-cookbook/configs/` 下的参数文件。

## 输入数据集格式

`recipes/grpo_gms8k_trio.py` 的 `--data-file` 接收 UTF-8 编码的 JSONL 文件：每个非空行是
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
  渲染后的提示词。其他字段可保留在数据行中。

已存在的 `--data-file` 会直接读取；文件不存在或指定 `--rebuild-data` 时，脚本会
下载 `openai/gsm8k` 并在该路径生成数据，因此不要对自定义文件使用
`--rebuild-data`。可先检查读取的记录数：

```bash
bash trio-cookbook/scripts/train_grpo_gms8k.sh \
  --data-file path/to/train.jsonl --prepare-only
```

`--prepare-only` 检查外层字段类型；对话消息或 token 列表的具体内容在训练时进一步
校验。

## gms8k 案例

该脚本使用 GSM8K 数据集运行 GRPO 训练。首次执行会将数据集缓存到
`.cache/huggingface/`，并生成供训练器读取的 `data/gsm8k_train.jsonl`。

### 启动训练

编辑 [grpo_gms8k.toml](../configs/grpo_gms8k.toml) 配置模型、数据路径、GRPO 超参数
和 SwanLab。默认配置运行 100 个训练 step，每个 batch 包含 4 个问题，每个问题
采样 8 个 completion，SwanLab 日志关闭。从仓库根目录运行：

```bash
bash trio-cookbook/scripts/train_grpo_gms8k.sh
```

临时试跑可通过启动脚本覆盖步数、batch 和 group 大小，其他超参数直接修改
TOML。`--config` 可以指定另一个参数文件；配置中的相对数据路径以仓库根目录为准。
若不在配置中设置 `grpo.save_every`，默认在最后一个训练 step 保存 checkpoint。

```bash
bash trio-cookbook/scripts/train_grpo_gms8k.sh \
  --steps 10 --batch-size 2 --group-size 4
```

`swanlab.mode` 可选 `disabled`、`online`、`local` 和 `offline`。可在配置中
设置 `workspace` 与 `tags`，也可临时通过命令行覆盖；`--tag`（别名 `--tags`）
可以重复使用：

```bash
bash trio-cookbook/scripts/train_grpo_gms8k.sh \
  --mode online --workspace my-organization --tag grpo --tag gsm8k
```

启用 SwanLab 后，训练结束时会以文本记录最终 checkpoint 地址，并展示最后
一批中最多 4 个问题的最终模型推理结果。最终步数、核心指标和 checkpoint
地址会在 SwanLab 结束前打印到终端，同时记录为 `trainer/final_summary` 文本。
需要重新下载并生成数据时，追加 `--rebuild-data`。查看脚本支持的覆盖参数：

```bash
bash trio-cookbook/scripts/train_grpo_gms8k.sh --help
```
