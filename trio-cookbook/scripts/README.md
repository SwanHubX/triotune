# 训练脚本

本目录存放训练启动脚本；训练案例实现位于
[grpo_gms8k_trio.py](../recipes/grpo_gms8k_trio.py)。
命令默认从仓库根目录执行，并使用 `uv` 管理项目环境。启动脚本会切换到仓库根目录，
再读取 `trio-cookbook/configs/` 下的参数文件。

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

启动脚本固定传入 `--rebuild-data`，每次都会从 `openai/gsm8k` 重新生成数据并
覆盖目标文件。使用自定义数据时，直接运行训练入口，不要传 `--rebuild-data`：

```bash
uv run python trio-cookbook/recipes/grpo_gms8k_trio.py \
  --config trio-cookbook/configs/grpo_gms8k.toml \
  --data-file path/to/train.jsonl --prepare-only
```

`--prepare-only` 检查外层字段类型；对话消息或 token 列表的具体内容在训练时进一步
校验。
已有的旧格式自定义数据文件需要将 `prompt` 字段改名为 `messages`。

## gms8k 案例

该脚本使用 GSM8K 数据集运行 GRPO 训练。首次执行会将数据集缓存到
`.cache/huggingface/`，并生成供训练器读取的 `data/gsm8k_train.jsonl`。

### 启动训练

编辑 [grpo_gms8k.toml](../configs/grpo_gms8k.toml) 配置模型、数据路径、GRPO 超参数
和 SwanLab。默认配置运行 100 个训练 step，每个 batch 包含 4 个问题，每个问题
采样 8 个 completion，SwanLab 日志为 `online` 模式。从仓库根目录运行：

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

`grpo.output_dir = true` 时，训练结束后会在终端打印最终 checkpoint 地址；
默认值为 `false`。启用 SwanLab 后，还会展示最后一批中最多 4 个问题的最终
模型推理结果。最终步数和核心指标会在 SwanLab 结束前打印到终端，
并记录为 `trainer/final_summary` 文本。
启动脚本每次都会重新生成数据。查看脚本支持的覆盖参数：

```bash
bash trio-cookbook/scripts/train_grpo_gms8k.sh --help
```

## 与独立 GRPO 案例对比

[grpo_gsm8k.py](../recipes/grpo_gsm8k.py) 可直接读取同一份
`data/gsm8k_train.jsonl`。若文件尚不存在，先用上面的启动脚本加 `--prepare-only`
准备；数据构建不在独立案例中进行。PyTRIO 训练工作空间由当前凭据决定，
`--workspace pytrio` 只指定 SwanLab 日志所属的工作空间，不改变 PyTRIO 凭据。

```bash
uv run python trio-cookbook/recipes/grpo_gsm8k.py \
  --data-file data/gsm8k_train.jsonl \
  --steps 100 --batch-size 4 --group-size 8 \
  --loss-fn importance_sampling \
  --workspace pytrio
```

两个入口均记录 `reward/mean`、`reward/std`、`trainer/loss`、
`policy/entropy`、`policy/kl_sample_train`、`optim/lr`、可用时的
`optim/grad_norm`，以及 `time/rollout`、`time/reward`、`time/train`、
`time/step`。KL 只统计非零 advantage 的 completion token；
`policy/entropy` 是采样 logprob 的负均值。数值受随机采样和远端服务耗时影响，
不保证两次运行完全相同；最后一步的 `time/step` 也因框架保存完整训练状态、
独立案例在循环后另存采样权重而不应直接比较。
