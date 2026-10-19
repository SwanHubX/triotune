# 训练脚本

## 脚本、配置与算法入口

脚本调用 recipe，recipe 读取 YAML 并使用对应算法包的公开入口：

| 启动脚本 | Recipe | 配置 | Python 入口 |
| --- | --- | --- | --- |
| [train_grpo_gms8k.sh](./train_grpo_gms8k.sh) | [train_grpo.py](../recipes/train_grpo.py) | [grpo_gms8k.yaml](../configs/grpo_gms8k.yaml) | `PPOConfig.from_modules` (`grpo`) |
| [train_ppo_gms8k.sh](./train_ppo_gms8k.sh) | [train_ppo.py](../recipes/train_ppo.py) | [ppo_gsm8k.yaml](../configs/ppo_gsm8k.yaml) | `PPOConfig.from_modules` (`ppo`) |
| [train_dpo_rlhf.sh](./train_dpo_rlhf.sh) | [train_dpo.py](../recipes/train_dpo.py) | [dpo_rlhf.yaml](../configs/dpo_rlhf.yaml) | `trio_cookbook.trainer.dpo` |
| [train_sft_huanhuan.sh](./train_sft_huanhuan.sh) | [train_sft.py](../recipes/train_sft.py) | [sft_huanhuan.yaml](../configs/sft_huanhuan.yaml) | `trio_cookbook.trainer.sft` |
| [train_opd_deepmath.sh](./train_opd_deepmath.sh) | [train_opd.py](../recipes/train_opd.py) | [opd_deepmath.yaml](../configs/opd_deepmath.yaml) | `trio_cookbook.trainer.opd` |

各配置的算法段（`grpo`、`dpo`、`sft`、`opd`）与 `swanlab`
合并为对应的 Config；`[model]` 与 `[data]` 由 recipe 读取。
内置配置统一使用 YAML；为兼容旧实验，仍可通过 `--config` 显式传入历史 TOML 文件。
训练器目录调整不影响脚本路径和配置字段，原有启动命令仍可使用。

PPO-family 配置使用模块化结构：`algorithm` 选择固定预设，`training` 管理训练
步数和批大小，`optimizer` 管理 Adam 参数，`rollout` 管理采样，`actor` 管理
策略损失，`critic` 管理 PPO value client，`swanlab` 管理记录。recipe 会将这些
模块映射到一个经过校验的 `PPOConfig`；例如将 `algorithm` 改为 `ppo` 并把
`critic.enabled` 设为 `true` 即可使用独立 critic。

如果需要 reference policy 的 KL 约束，在 `actor.kl_ctl` 设置正数，并在
`reference.path` 指定模型（支持 `${model.name}` 这类 YAML 引用）。省略
`reference.path` 时，训练器使用学生的初始 base model 创建冻结 sampling client；
`actor.kl_ctl: 0` 时不会创建或请求 reference 打分。
DPO 脚本保留 `uv run --extra dpo`，用于安装本地损失计算需要的 PyTorch。

GRPO 由统一的 PPO trainer 执行；独立版 recipe 仍可直接调用 PyTRIO。
OPD 另有 `[teacher]` 配置教师模型，
`[runtime].timeout` 设置 PyTRIO 与数据下载超时。

## 输入数据集格式

### GRPO

`train_grpo.py` 的 `--data-file` 接收 UTF-8 JSONL，每行一条记录：

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

`train_dpo.py` 的 `[data].file`（或 `--data-file`）接收 UTF-8 JSONL，
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


以下命令默认从仓库根目录执行，启动脚本会切换到仓库根目录并使用 `uv`。

## HH-RLHF DPO 案例

数据集来自 ModelScope 的 `Anthropic/hh-rlhf`，具体参数配置见
[dpo_rlhf.yaml](../configs/dpo_rlhf.yaml)。

预处理数据：

```bash
bash trio-cookbook/scripts/train_dpo_rlhf.sh --prepare-only
```

启动训练：

```bash
bash trio-cookbook/scripts/train_dpo_rlhf.sh
```

## GSM8K GRPO 案例

数据集为 GSM8K，具体配置见 [grpo_gms8k.yaml](../configs/grpo_gms8k.yaml)。

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

## DeepMath OPD 异步案例

准备和检查数据，不创建 PyTRIO 客户端：

```bash
bash trio-cookbook/scripts/train_opd_deepmath.sh --prepare-only
```

启动默认训练，或按独立案例的命令覆盖参数：

```bash
bash trio-cookbook/scripts/train_opd_deepmath.sh

bash trio-cookbook/scripts/train_opd_deepmath.sh \
  --steps 100 --batch-size 4 --group-size 4 --max-tokens 512 \
  --sample-size 5000 --base-model Qwen/Qwen3.5-4B \
  --teacher-base-model Qwen/Qwen3.8-27B

# 完整数据、关闭 SwanLab；已有分片会直接复用
bash trio-cookbook/scripts/train_opd_deepmath.sh \
  --num-shards 10 --sample-size 0 --no-swanlab

bash trio-cookbook/scripts/train_opd_deepmath.sh --help
```
