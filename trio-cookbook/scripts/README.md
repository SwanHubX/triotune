# 训练脚本

## 脚本、配置与算法入口

脚本调用 recipe，recipe 读取 TOML 并使用对应算法包的公开入口：

| 启动脚本 | Recipe | 配置 | Python 入口 |
| --- | --- | --- | --- |
| [train_grpo_gms8k.sh](./train_grpo_gms8k.sh) | [grpo_gms8k_trio.py](../recipes/grpo_gms8k_trio.py) | [grpo_gms8k.toml](../configs/grpo_gms8k.toml) | `trio_cookbook.trainer.grpo` |
| [train_dpo_rlhf.sh](./train_dpo_rlhf.sh) | [dpo_rlhf_trio.py](../recipes/dpo_rlhf_trio.py) | [dpo_rlhf.toml](../configs/dpo_rlhf.toml) | `trio_cookbook.trainer.dpo` |
| [train_sft_huanhuan.sh](./train_sft_huanhuan.sh) | [sft_huanhuan_trio.py](../recipes/sft_huanhuan_trio.py) | [sft_huanhuan.toml](../configs/sft_huanhuan.toml) | `trio_cookbook.trainer.sft` |

各配置的算法表（`[grpo]`、`[dpo]`、`[sft]`）与 `[swanlab]`
合并为对应的 Config；`[model]` 与 `[data]` 由 recipe 读取。
训练器目录调整不影响脚本路径和配置字段，原有启动命令仍可使用。
DPO 脚本保留 `uv run --extra dpo`，用于安装本地损失计算需要的 PyTorch。

数独框架示例 [grpo_sudoku_trio.py](../recipes/grpo_sudoku_trio.py)
同样使用 `trio_cookbook.trainer.grpo`，通过自己的命令行参数配置。
独立版 recipe 直接调用 PyTRIO。OPD 当前提供算法模块和使用示例，尚无启动脚本或配置 recipe。

## 输入数据集格式

### GRPO

`grpo_gms8k_trio.py` 的 `--data-file` 接收 UTF-8 JSONL，每行一条记录：

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

### DPO

`dpo_rlhf_trio.py` 的 `[data].file`（或 `--data-file`）接收 UTF-8 JSONL，
每行一组偏好数据：

```jsonl
{"messages":[{"role":"user","content":"你好"}],"chosen":[{"role":"assistant","content":"你好！"}],"rejected":[{"role":"assistant","content":"不想回答。"}]}
```

`messages` 是共同的对话前文；`chosen` 和 `rejected` 分别是更好和更差的回复，
三者均为必填。预处理脚本生成的回复字段各含一条 assistant 消息。

### SFT

`SFTTrainer` 的每条样本可使用以下一种格式，例如 `{"text":"你好，世界！"}`：

- `text`（或 `dataset_text_field` 指定的字段）：字符串。
- `messages`：非空聊天消息列表，每条包含非空字符串 `role` 和字符串 `content`。
- `prompt` 与 `completion`：两者同为字符串或聊天消息列表；默认只对 completion
  计算损失。
- `input_ids`：非空的非负整数 token 列表；可附等长 `labels`，其中 `-100`
  表示不参与损失。

截断后须保留至少一个有效的 next-token 标签；`assistant_only_loss=True` 要求对话数据
及支持 assistant token mask 的聊天模板。

嬛嬛示例脚本读取 `data/huanhuan.json` 中的 `instruction`、`input`、`output`，
再转换为 `prompt`/`completion`；上述行格式适用于直接向 `SFTTrainer` 传入数据。

以下命令默认从仓库根目录执行，启动脚本会切换到仓库根目录并使用 `uv`。

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

## 嬛嬛 SFT 案例

训练数据下载：

```bash
mkdir -p data
curl --fail --location https://raw.githubusercontent.com/KMnO4-zx/huanhuan-chat/2d6c72909d6c3b35e3435cd7ce5658306737bfbd/dataset/train/lora/huanhuan.json --output data/huanhuan.json
```

训练启动：

```bash
bash trio-cookbook/scripts/train_sft_huanhuan.sh
```
