# PyTRIO 训练器

本目录提供基于 PyTRIO 客户端的纯文本 GRPO、DPO、SFT、OPD 训练器及运行示例。
OPD 的算法说明见 [训练器说明](trainer/README.md#opd-训练)，DeepMath 案例见
[脚本说明](scripts/README.md#deepmath-opd-异步案例)。源码目录名为
`trio-cookbook`，Python 导入名为 `trio_cookbook`。

## 目录结构

```text
trio-cookbook/
├── configs/          # GRPO、DPO、SFT、OPD 训练参数
├── recipes/          # 框架版和独立版训练入口
├── scripts/          # 启动脚本及使用说明
└── trainer/          # 按算法组织的训练实现和稳定导入入口
│   ├── common/       # 配置、client、验证、日志与 checkpoint 公共契约
│   ├── grpo/         # 分组采样、奖励、优势、Datum 与训练编排
│   ├── dpo/          # 偏好数据、成对损失与训练编排
│   ├── sft/          # 标签处理与异步流水线训练
│   └── opd/          # 教师评分、蒸馏信号与训练编排
```

训练器模块的详细职责见 [trainer/README.md](trainer/README.md)；脚本参数及独立版示例见
[scripts/README.md](scripts/README.md)。

## 数据格式

### GRPO

GSM8K 训练数据使用 JSONL，每行包含 `messages` 和 `answer`：

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

DPO 训练数据使用 JSONL，每行包含一组偏好回复：

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

### OPD

DeepMath OPD 从 ModelScope 的 `AI-ModelScope/DeepMath-103K` 下载 train parquet，
只将 `question` 转换为带题目后缀的 user 消息，不使用参考答案或 R1 解答作为标签。
样本以固定 seed 打乱后抽样；框架按配置的步数循环使用这些 prompt。

## 启动命令

在仓库根目录运行（启动脚本使用 `uv`）：

### GRPO

```bash
# 准备并检查数据，不启动训练
bash trio-cookbook/scripts/train_grpo_gms8k.sh --prepare-only

# 按 configs/grpo_gms8k.toml 启动训练
bash trio-cookbook/scripts/train_grpo_gms8k.sh
```

### DPO

```bash
# 准备并检查数据，不启动训练
bash trio-cookbook/scripts/train_dpo_rlhf.sh --prepare-only

# 按 configs/dpo_rlhf.toml 启动训练
bash trio-cookbook/scripts/train_dpo_rlhf.sh
```

### SFT

按 [脚本说明](scripts/README.md) 下载数据后，使用 `configs/sft_huanhuan.toml` 启动训练：

```bash
bash trio-cookbook/scripts/train_sft_huanhuan.sh
```

### OPD

```bash
# 下载、抽样并检查问题，不创建训练客户端
bash trio-cookbook/scripts/train_opd_deepmath.sh --prepare-only

# 按 configs/opd_deepmath.toml 异步训练
bash trio-cookbook/scripts/train_opd_deepmath.sh
```
