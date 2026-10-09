# 训练脚本

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

训练数据来自 [huanhuan-chat](https://github.com/KMnO4-zx/huanhuan-chat)，存放在本地 `data/`（该目录不会提交）。先从原项目固定版本下载：

```bash
mkdir -p data
curl --fail --location https://raw.githubusercontent.com/KMnO4-zx/huanhuan-chat/2d6c72909d6c3b35e3435cd7ce5658306737bfbd/dataset/train/lora/huanhuan.json --output data/huanhuan.json
```

配置见 [sft_huanhuan.toml](../configs/sft_huanhuan.toml)。默认使用前 100 条数据、batch size 1、100 step，并记录到 SwanLab：

`[swanlab]` 中可设置 `workspace`、`tags`、`run_name` 和 `log_dir`，分别指定工作区、标签、运行名称和本地日志目录。

```bash
bash trio-cookbook/scripts/train_sft_huanhuan.sh
```

独立 PyTRIO 版本可用同样参数作对照：

```bash
uv run python trio-cookbook/recipes/sft_huanhuan.py --steps 100 --batch-size 1 --limit 100 --epochs 1 --save-every 0 --run-name huanhuan-direct-100step-seed42 --swanlab
```

当前 PyTRIO 服务只支持 Qwen 3.5 系列，因此这个案例沿用原项目的对话数据和角色提示，基础模型改为配置中的 Qwen 3.5 4B。
