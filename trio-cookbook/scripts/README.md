# 训练脚本

## 输入数据集格式

`recipes/grpo_gms8k_trio.py` 的 `--data-file` 接收 UTF-8 编码的 JSONL 文件：每个非空行是
一条 JSON 对象，文件至少包含一条有效记录。例如：

```jsonl
{"messages":[{"role":"user","content":"What is 6 times 7? Answer with \\boxed{}."}],"answer":"42","question":"What is 6 times 7?"}
```

- `messages`（必填）：字符串、非空的整数 token ID 列表，或非空的对话消息列表；
  对话消息的 `role` 和 `content` 应为字符串。
- `answer`（必填）：字符串形式的参考数值答案，例如 `"42"`。奖励函数提取生成文本
  中最后一个 `\boxed{}` 的内容，去掉千位逗号、首尾空白和末尾句点后比较；提示词
  应要求模型用 `\boxed{}` 作答。
- `question`（可选）：原始问题文本，用作 SwanLab 最终推理样例的标题；省略时使用
  渲染后的提示词。其他字段可保留在数据行中。

`--prepare-only` 检查外层字段类型；对话消息或 token 列表的具体内容在训练时进一步
校验。


## HH-RLHF DPO 案例

数据集来自 ModelScope 的 `Anthropic/hh-rlhf`，具体参数配置见
[dpo_rlhf.toml](../configs/dpo_rlhf.toml)。

预处理数据：

```bash
bash trio-cookbook/scripts/train_dpo_rlhf.sh --prepare-only
```

启动训练：

```bash
bash trio-cookbook/scripts/train_dpo_rlhf.sh
```

本目录存放训练启动脚本；以下为
[grpo_gms8k_trio.py](../recipes/grpo_gms8k_trio.py) 的使用说明。
命令默认从仓库根目录执行，并使用 `uv` 管理项目环境。启动脚本会切换到仓库根目录，
再读取 `trio-cookbook/configs/` 下的参数文件。

## GSM8K GRPO 案例

数据集为 GSM8K，具体配置见 [grpo_gms8k.toml](../configs/grpo_gms8k.toml)。

预处理数据：

```bash
bash trio-cookbook/scripts/train_grpo_gms8k.sh --prepare-only
```

训练启动：

```bash
bash trio-cookbook/scripts/train_grpo_gms8k.sh
```

查看脚本支持的覆盖参数：

```bash
bash trio-cookbook/scripts/train_grpo_gms8k.sh --help
```
