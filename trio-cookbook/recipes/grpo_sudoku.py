"""使用 PyTRIO 在 4x4 数独任务上进行同步 GRPO 训练。

本脚本改写自 ``sudoku_trl_grpo/train_grpo.py``，训练循环沿用
``trio-cookbook/recipes/grpo_gsm8k.py`` 的 PyTRIO 写法：

1. 读取并渲染 4x4 数独题目；
2. 用当前 LoRA 权重为每道题采样一组回答；
3. 根据格式、长度、原题线索、行、列和宫格规则计算奖励；
4. 在同一题的回答组内计算 ``advantage = reward - mean(reward)``；
5. 构造带旧策略 logprobs 的 PyTRIO Datum，执行一次策略更新。

最小试跑（使用仓库 ``tests/sudoku_4x4_qa.jsonl`` 数据）：

    uv run python trio-cookbook/recipes/grpo_sudoku.py \
        --steps 1 \
        --batch-size 1 \
        --group-size 4 \
        --swanlab-mode disabled

正式训练：

    uv run python trio-cookbook/recipes/grpo_sudoku.py \
        --steps 100 \
        --batch-size 2 \
        --group-size 6 \
        --max-tokens 1648 \
        --seed 42 \
        --loss-fn importance_sampling \
        --swanlab-mode online
"""

from __future__ import annotations

import argparse
import math
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pytrio as trio
import swanlab
from datasets import Dataset, load_dataset
from tqdm import tqdm


LOSS_FNS = ("importance_sampling", "ppo")
ANSWER_PATTERN = re.compile(
    r"^<think>[\s\S]*?</think>\s*<answer>([\s\S]*?)</answer>$"
)
SUDOKU_FORMAT = "[[{},{},{},{}],\n[{},{},{},{}],\n[{},{},{},{}],\n[{},{},{},{}]]"
PROMPT_TEMPLATE = """请作为数独专家完成4x4数独题目。数独题目中已给出数字为限定条件，“_”表示待填入1-4的空白位置。
## 数独规则
- 根据数独矩阵中已给出的数字推测“_”处的数字
- 每行：数字1-4各出现一次
- 每列：数字1-4各出现一次
- 每个2×2宫格：数字1-4各出现一次
## 解题流程
1. 选择一处空白位置“_”，按行、列、宫格分析应该填入的数字
2. 打印填补一处空白的矩阵，判断该次填充是否正确
3. 重复流程1，直至所有空白位置“_”被填完。
## 输出格式
输出格式为<think>...</think>\n<answer>...</answer>
在<think>...</think>中填入思考过程
在<answer>...</answer>中填写python List格式的最终矩阵
输出格式案例：
<think>
详细推理过程，包括：
- 一步步的思考过程
- 对题目的复述，以及要分析的空白位置
- 填入空白位置的数字以及理由，按行、列、宫格规则分析
- 检查填入后的数独矩阵，确保符合数独规则，确保没有修改题目限定条件
</think>
<answer>[[1,2,3,4],[4,3,...],...]</answer>
## 数独题目：
{}"""
COMPARISON_METRIC_KEYS = (
    "reward/mean",
    "reward/std",
    "reward/format/mean",
    "reward/length/mean",
    "reward/clue/mean",
    "reward/row/mean",
    "reward/column/mean",
    "reward/block/mean",
    "reward/exact/mean",
    "policy/entropy",
    "policy/kl_sample_train",
    "rollout/completion_length_mean",
    "rollout/completion_length_min",
    "rollout/completion_length_max",
    "rollout/degenerate_fraction",
    "trainer/loss",
    "trainer/tokens",
    "trainer/num_datums",
    "trainer/skipped_groups",
    "optim/lr",
    "optim/grad_norm",
)


@dataclass(slots=True)
class GRPOConfig:
    """命令行参数解析后的训练配置。"""

    base_model: str
    data_file: Path
    difficulty: str
    lora_rank: int
    steps: int
    all_data: bool
    batch_size: int
    group_size: int
    max_tokens: int
    temperature: float
    top_p: float
    seed: int
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    grad_clip_norm: float
    loss_fn: str
    swanlab_mode: str
    swanlab_project: str
    run_name: str | None
    print_samples: bool


@dataclass(slots=True)
class RewardBreakdown:
    """一条回答的六项奖励，以及只用于监控的精确答案指标。"""

    format: float
    length: float
    clue: float
    row: float
    column: float
    block: float
    exact: float

    @property
    def total(self) -> float:
        return self.format + self.length + self.clue + self.row + self.column + self.block

    def reward_metrics(self) -> dict[str, float]:
        return {
            "reward/format/mean": self.format,
            "reward/length/mean": self.length,
            "reward/clue/mean": self.clue,
            "reward/row/mean": self.row,
            "reward/column/mean": self.column,
            "reward/block/mean": self.block,
            "reward/exact/mean": self.exact,
        }


@dataclass(slots=True)
class RolloutSample:
    """一条回答及 GRPO 更新需要的旧策略概率和组内 advantage。"""

    tokens: list[int]
    logprobs: list[float]
    text: str
    rewards: RewardBreakdown
    advantage: float

    @property
    def reward(self) -> float:
        return self.rewards.total


def default_data_file() -> Path:
    """返回原数独项目的数据生成脚本所使用的默认路径。"""

    repository_root = Path(__file__).resolve().parents[2]
    return repository_root / "tests/sudoku_4x4_qa.jsonl"


def parse_args() -> GRPOConfig:
    """解析并验证训练参数。"""

    parser = argparse.ArgumentParser(description="PyTRIO 同步版 GRPO / 4x4 Sudoku")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B", help="PyTRIO 基础模型名")
    parser.add_argument("--data-file", type=Path, default=default_data_file(), help="数独 JSONL 数据")
    parser.add_argument(
        "--difficulty",
        choices=("simple", "medium", "difficult", "all"),
        default="simple",
        help="保留的数据难度",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--steps", type=int, default=500, help="GRPO 优化步数")
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="遍历过滤后的训练集一次；启用后忽略 --steps",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="每步使用的数独题目数")
    parser.add_argument("--group-size", type=int, default=6, help="每道题采样的回答数")
    parser.add_argument("--max-tokens", type=int, default=1648, help="每条回答最多生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样 temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="采样 top_p")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Adam learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--grad-clip-norm", type=float, default=0.2, help="梯度裁剪阈值")
    parser.add_argument(
        "--loss-fn",
        choices=LOSS_FNS,
        default="importance_sampling",
        help="PyTRIO 内置策略损失",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "disabled"),
        default="online",
        help="SwanLab 记录模式",
    )
    parser.add_argument("--swanlab-project", default="sudoku-grpo-pytrio", help="SwanLab project")
    parser.add_argument("--run-name", default=None, help="SwanLab run 和最终权重名称")
    parser.add_argument(
        "--print-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="每步打印第一条模型回答",
    )
    args = parser.parse_args()

    positive_values = {
        "lora_rank": args.lora_rank,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
        "max_tokens": args.max_tokens,
        "learning_rate": args.learning_rate,
        "eps": args.eps,
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    if args.group_size < 2:
        parser.error("--group-size 至少为 2，GRPO 才能计算组内相对 advantage")
    if not 0.0 <= args.top_p <= 1.0:
        parser.error("--top-p 必须位于 [0, 1]")
    if args.temperature < 0.0:
        parser.error("--temperature 不能小于 0")

    return GRPOConfig(
        base_model=args.base_model,
        data_file=args.data_file.expanduser().resolve(),
        difficulty=args.difficulty,
        lora_rank=args.lora_rank,
        steps=args.steps,
        all_data=args.all_data,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        loss_fn=args.loss_fn,
        swanlab_mode=args.swanlab_mode,
        swanlab_project=args.swanlab_project,
        run_name=args.run_name,
        print_samples=args.print_samples,
    )


def validate_row(row: dict[str, Any], index: int) -> None:
    """在发起远程训练前检查一条数独数据是否完整。"""

    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(question, str) or len(question) != 16:
        raise ValueError(f"Row {index}: question 必须是长度为 16 的字符串")
    if any(char not in "1234_" for char in question):
        raise ValueError(f"Row {index}: question 只能包含 1-4 和下划线")
    if not isinstance(answer, str) or len(answer) != 16 or any(char not in "1234" for char in answer):
        raise ValueError(f"Row {index}: answer 必须是由 1-4 组成的长度为 16 的字符串")
    if any(given != "_" and given != solved for given, solved in zip(question, answer, strict=True)):
        raise ValueError(f"Row {index}: answer 修改了 question 中的已知数字")


def load_sudoku_train(config: GRPOConfig) -> Dataset:
    """加载 JSONL，按难度过滤、打乱，并复现原任务的 90% 训练切分。"""

    if not config.data_file.is_file():
        raise FileNotFoundError(
            f"找不到数独数据：{config.data_file}\n"
            "请确认仓库 tests/sudoku_4x4_qa.jsonl 存在，或通过 --data-file 指定已有 JSONL。"
        )
    dataset = load_dataset("json", data_files=str(config.data_file), split="train")
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset)!r}")
    required_columns = {"question", "answer", "label"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise ValueError(f"数独数据缺少字段：{sorted(missing)}")
    if config.difficulty != "all":
        dataset = dataset.filter(lambda row: row["label"] == config.difficulty)
    if len(dataset) < 2:
        raise ValueError(f"难度 {config.difficulty!r} 过滤后至少需要 2 条数据")
    dataset = dataset.shuffle(seed=config.seed)
    dataset = dataset.train_test_split(test_size=0.1, seed=config.seed)["train"]
    for index, row in enumerate(dataset):
        validate_row(dict(row), index)
    return dataset


def format_question(question: str) -> str:
    """把紧凑的 16 字符题面渲染成 4x4 Python List。"""

    return SUDOKU_FORMAT.format(*question)


def _token_ids(value: Any, source: str) -> list[int]:
    """把 tokenizer 的多种返回类型规范化为一维 token ID。"""

    if isinstance(value, Mapping):
        value = value.get("input_ids")
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    elif hasattr(value, "ids"):
        value = value.ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value or not all(
        isinstance(token, Integral) and not isinstance(token, bool) for token in value
    ):
        raise ValueError(f"{source} must return a non-empty one-dimensional token sequence")
    return [int(token) for token in value]


def build_prompt(tokenizer: Any, question: str) -> list[int]:
    """构造以 assistant 头结尾、由模型自行生成 ``<think>`` 的输入。"""

    messages = [
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(format_question(question)),
        }
    ]
    prompt_tokens = _token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        ),
        "tokenizer.apply_chat_template",
    )
    thinking_prefix = _token_ids(
        tokenizer.encode("<think>\n", add_special_tokens=False),
        "tokenizer.encode",
    )
    if prompt_tokens[-len(thinking_prefix) :] != thinking_prefix:
        raise ValueError("Chat template must end with the Qwen <think> generation prefix")
    prompt_tokens = prompt_tokens[: -len(thinking_prefix)]
    if not prompt_tokens:
        raise ValueError("Chat template became empty after removing the thinking prefix")
    return prompt_tokens


def extract_board(response: str) -> list[int] | None:
    """从严格的 think/answer 输出中提取 16 个数字。"""

    match = ANSWER_PATTERN.match(response)
    if match is None:
        return None
    digits = re.findall(r"\d+", match.group(1))
    if len(digits) != 16:
        return None
    try:
        return [int(value) for value in digits]
    except ValueError:
        return None


def length_reward(response: str) -> float:
    """复现原任务的长度奖励，最大值为 0.4。"""

    length = len(response)
    if length < 1584:
        return 0.4
    return 0.4 * max(0.0, (1648 - length) / 64)


def uniqueness_score(groups: list[list[int]]) -> float:
    """返回若干行、列或宫格的平均唯一数字比例。"""

    return sum(len(set(group)) / 4 for group in groups) / len(groups)


def score_sudoku_response(
    response: str,
    question: str,
    ground_truth: str,
) -> RewardBreakdown:
    """计算原数独任务的六项加权奖励，总分最高为 3.0。"""

    board = extract_board(response)
    length = length_reward(response)
    if board is None:
        return RewardBreakdown(0.0, length, 0.0, 0.0, 0.0, 0.0, 0.0)

    in_range = all(1 <= value <= 4 for value in board)
    format_score = 0.6 if in_range else 0.0

    given_indices = [index for index, value in enumerate(question) if value != "_"]
    clue_matches = sum(board[index] == int(question[index]) for index in given_indices)
    clue_score = clue_matches / len(given_indices) if given_indices else 0.0

    rows = [board[start : start + 4] for start in range(0, 16, 4)]
    columns = [[board[row * 4 + column] for row in range(4)] for column in range(4)]
    blocks = [
        [
            board[(block_row + row) * 4 + block_column + column]
            for row in range(2)
            for column in range(2)
        ]
        for block_row in (0, 2)
        for block_column in (0, 2)
    ]
    row_score = 0.4 * uniqueness_score(rows)
    column_score = 0.3 * uniqueness_score(columns)
    block_score = 0.3 * uniqueness_score(blocks)
    exact = float(board == [int(value) for value in ground_truth])
    return RewardBreakdown(
        format_score,
        length,
        clue_score,
        row_score,
        column_score,
        block_score,
        exact,
    )


def get_stop_sequences(tokenizer: Any) -> list[str]:
    """组合 tokenizer EOS 和 Qwen chat 结束符，并去除空值和重复值。"""

    candidates = [getattr(tokenizer, "eos_token", None), "<|im_end|>"]
    return list(dict.fromkeys(value for value in candidates if isinstance(value, str) and value))


def run_rollout_group(
    sampling_client: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    question: str,
    ground_truth: str,
    sampling_params: trio.SamplingParams,
    group_size: int,
) -> list[RolloutSample]:
    """为一道题采样多个回答、评分并计算组内中心化 advantage。"""

    result = sampling_client.sample(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=group_size,
        sampling_params=sampling_params,
        return_text=True,
    ).result()
    if len(result.sequences) != group_size:
        raise ValueError(f"Expected {group_size} samples, got {len(result.sequences)}")
    raw_samples: list[tuple[list[int], list[float], str, RewardBreakdown]] = []
    for sequence in result.sequences:
        tokens = list(sequence.tokens)
        logprobs = [float(value) for value in sequence.logprobs]
        if not tokens:
            raise ValueError("PyTRIO returned an empty completion")
        if len(tokens) != len(logprobs):
            raise ValueError(
                f"Generated token/logprob length mismatch: {len(tokens)} != {len(logprobs)}"
            )
        text = sequence.text
        if text is None:
            text = tokenizer.decode(tokens, skip_special_tokens=True)
        rewards = score_sudoku_response(text, question, ground_truth)
        raw_samples.append((tokens, logprobs, text, rewards))

    if len(raw_samples) < 2:
        raise ValueError("GRPO rollout group must contain at least two samples")
    mean_reward = sum(item[3].total for item in raw_samples) / len(raw_samples)
    return [
        RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            text=text,
            rewards=rewards,
            advantage=rewards.total - mean_reward,
        )
        for tokens, logprobs, text, rewards in raw_samples
    ]


def build_grpo_datum(prompt_tokens: list[int], sample: RolloutSample) -> trio.Datum:
    """构造 PyTRIO importance_sampling/ppo 所需的单 token 位移 Datum。"""

    observation_len = len(prompt_tokens) - 1
    input_tokens = prompt_tokens + sample.tokens[:-1]
    target_tokens = [0] * observation_len + sample.tokens
    old_logprobs = [0.0] * observation_len + sample.logprobs
    advantages = [0.0] * observation_len + [sample.advantage] * len(sample.tokens)
    if not len(input_tokens) == len(target_tokens) == len(old_logprobs) == len(advantages):
        raise ValueError("GRPO datum fields must have the same token length")
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(old_logprobs, dtype=np.float32),
            "advantages": np.asarray(advantages, dtype=np.float32),
        },
    )


def run_optimization_step(
    training_client: Any,
    datums: list[trio.Datum],
    *,
    loss_fn: str,
    loss_fn_config: dict[str, Any],
    adam_params: trio.AdamParams,
) -> dict[str, Any]:
    """反向传播成功后才执行优化器，避免错误地应用残留梯度。"""

    if not datums:
        return {}
    forward_result = training_client.forward_backward(
        datums,
        loss_fn=loss_fn,
        loss_fn_config=loss_fn_config,
    ).result()
    optimizer_result = training_client.optim_step(adam_params).result()
    metrics = dict(forward_result.metrics)
    optimizer_metrics = getattr(optimizer_result, "metrics", None)
    if isinstance(optimizer_metrics, Mapping):
        metrics.update(optimizer_metrics)
    kl = sample_train_kl(datums, forward_result)
    if kl is not None:
        metrics["policy/kl_sample_train"] = kl
    return metrics


def sample_train_kl(datums: list[trio.Datum], forward_result: Any) -> float | None:
    """按框架相同口径，仅在非零 advantage token 上估计采样/训练 KL。"""

    outputs = getattr(forward_result, "loss_fn_outputs", None)
    if not isinstance(outputs, list) or len(outputs) != len(datums):
        return None
    differences: list[float] = []
    for datum, output in zip(datums, outputs, strict=True):
        if not isinstance(output, Mapping) or "logprobs" not in output:
            return None
        old_logprobs = getattr(datum.loss_fn_inputs.get("logprobs"), "data", None)
        advantages = getattr(datum.loss_fn_inputs.get("advantages"), "data", None)
        current_logprobs = getattr(output["logprobs"], "data", None)
        if not all(isinstance(values, list) for values in (old_logprobs, advantages, current_logprobs)):
            return None
        if not len(old_logprobs) == len(advantages) == len(current_logprobs):
            return None
        differences.extend(
            (float(old) - float(current)) ** 2
            for old, advantage, current in zip(
                old_logprobs,
                advantages,
                current_logprobs,
                strict=True,
            )
            if float(advantage) != 0.0
        )
    return 0.5 * sum(differences) / len(differences) if differences else None


def get_num_steps(dataset: Dataset, config: GRPOConfig) -> int:
    """计算训练步数；all-data 模式保证每条样本最多使用一次。"""

    if config.all_data:
        return math.ceil(len(dataset) / config.batch_size)
    return min(config.steps, math.ceil(len(dataset) / config.batch_size))


def pick_batch(dataset: Dataset, step: int, batch_size: int) -> Dataset:
    """按照打乱后的固定顺序选择当前 batch。"""

    start = step * batch_size
    indices = list(range(start, min(start + batch_size, len(dataset))))
    return dataset.select(indices)


def model_slug(base_model: str) -> str:
    """把模型名转换成可用作权重名称的短字符串。"""

    name = base_model.rsplit("/", 1)[-1].lower().replace("qwen3.5", "qwen35")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def build_run_name(config: GRPOConfig, effective_steps: int) -> str:
    """生成 SwanLab run 和最终 LoRA 权重共用的名称。"""

    if config.run_name:
        return config.run_name
    loss_name = config.loss_fn.replace("_", "-")
    return f"grpo-sync-{model_slug(config.base_model)}-sudoku-{loss_name}-steps{effective_steps}"


def init_swanlab_run(
    config: GRPOConfig,
    effective_steps: int,
    dataset_size: int,
    run_name: str,
) -> Any | None:
    """按配置初始化 SwanLab。"""

    if config.swanlab_mode == "disabled":
        return None
    serialized_config = asdict(config)
    serialized_config["data_file"] = str(config.data_file)
    serialized_config.update(
        {
            "effective_steps": effective_steps,
            "dataset_size": dataset_size,
            "weights_name": run_name,
        }
    )
    return swanlab.init(
        project=config.swanlab_project,
        name=run_name,
        mode=config.swanlab_mode,
        config=serialized_config,
    )


def mean_reward_metrics(samples: list[RolloutSample]) -> dict[str, float]:
    """汇总一次 step 中各奖励分量的平均值。"""

    if not samples:
        return {}
    metric_rows = [sample.rewards.reward_metrics() for sample in samples]
    return {
        key: sum(row[key] for row in metric_rows) / len(metric_rows)
        for key in metric_rows[0]
    }


def numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """过滤 PyTRIO 返回指标中的非数字及非有限值。"""

    return {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float) and math.isfinite(float(value))
    }


def count_train_tokens(datums: list[trio.Datum]) -> int:
    """统计 advantage 非零、实际参与策略损失的 completion token 数。"""

    return sum(
        1
        for datum in datums
        for value in datum.loss_fn_inputs["advantages"].data
        if float(value) != 0.0
    )


def main(config: GRPOConfig) -> None:
    """运行同步 PyTRIO GRPO 训练循环并保存最终 LoRA 权重。"""

    np.random.seed(config.seed)
    print(f"Loading Sudoku dataset: {config.data_file}")
    train_data = load_sudoku_train(config)
    effective_steps = get_num_steps(train_data, config)
    run_name = build_run_name(config, effective_steps)
    print(f"Loaded {len(train_data)} training examples; run name: {run_name}")

    service_client = trio.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
        rank=config.lora_rank,
        seed=config.seed,
    )
    tokenizer = training_client.get_tokenizer()
    sampling_params = trio.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        stop=get_stop_sequences(tokenizer),
    )
    adam_params = trio.AdamParams(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.eps,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
    )
    swanlab_run = init_swanlab_run(config, effective_steps, len(train_data), run_name)

    try:
        for step in range(effective_steps):
            batch = pick_batch(train_data, step, config.batch_size)
            sampling_client = training_client.save_weights_and_get_sampling_client()
            datums: list[trio.Datum] = []
            step_samples: list[RolloutSample] = []
            completion_lengths: list[int] = []
            degenerate_groups = 0

            for row in tqdm(batch, desc=f"GRPO step {step + 1}", unit="prompt"):
                prompt_tokens = build_prompt(tokenizer, row["question"])
                samples = run_rollout_group(
                    sampling_client=sampling_client,
                    tokenizer=tokenizer,
                    prompt_tokens=prompt_tokens,
                    question=row["question"],
                    ground_truth=row["answer"],
                    sampling_params=sampling_params,
                    group_size=config.group_size,
                )
                step_samples.extend(samples)
                completion_lengths.extend(len(sample.tokens) for sample in samples)
                if config.print_samples:
                    tqdm.write(
                        "\n"
                        + "#" * 20
                        + " 推理案例 "
                        + "#" * 20
                        + f"\n数独题目：\n{format_question(row['question'])}"
                        + f"\n模型的第一个回答：\n{samples[0].text}"
                    )
                if all(sample.advantage == 0.0 for sample in samples):
                    degenerate_groups += 1
                    continue
                datums.extend(build_grpo_datum(prompt_tokens, sample) for sample in samples)

            raw_loss_metrics = run_optimization_step(
                training_client,
                datums,
                loss_fn=config.loss_fn,
                loss_fn_config={},
                adam_params=adam_params,
            )
            loss_metrics = numeric_metrics(raw_loss_metrics)
            reward_metrics = mean_reward_metrics(step_samples)
            rewards = [sample.reward for sample in step_samples]
            mean_reward = sum(rewards) / len(rewards)
            reward_std = math.sqrt(
                sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
            )
            avg_completion_length = sum(completion_lengths) / len(completion_lengths)
            degenerate_fraction = degenerate_groups / len(batch)
            train_tokens = count_train_tokens(datums)
            metrics = {
                "reward/mean": mean_reward,
                "reward/std": reward_std,
                **reward_metrics,
                "rollout/degenerate_fraction": degenerate_fraction,
                "rollout/completion_length_mean": avg_completion_length,
                "rollout/completion_length_min": float(min(completion_lengths)),
                "rollout/completion_length_max": float(max(completion_lengths)),
                "trainer/num_datums": float(len(datums)),
                "trainer/skipped_groups": float(degenerate_groups),
                "trainer/tokens": float(train_tokens),
                "optim/lr": config.learning_rate,
            }
            logprobs = [value for sample in step_samples for value in sample.logprobs]
            if logprobs:
                metrics["policy/entropy"] = -sum(logprobs) / len(logprobs)
            for key, value in loss_metrics.items():
                if isinstance(value, int | float) and math.isfinite(float(value)):
                    metrics[str(key)] = float(value)
            for loss_key in ("loss_mean", "loss"):
                if loss_key in loss_metrics:
                    metrics["trainer/loss"] = float(loss_metrics[loss_key])
                    break
            for grad_norm_key in ("optim/grad_norm", "grad_norm", "gradient_norm"):
                if grad_norm_key in loss_metrics:
                    metrics["optim/grad_norm"] = float(loss_metrics[grad_norm_key])
                    break
            if swanlab_run is not None:
                swanlab_run.log(
                    {key: metrics[key] for key in COMPARISON_METRIC_KEYS if key in metrics}
                    | {"trainer/step": float(step + 1)},
                    step=step + 1,
                )

            loss_text = (
                f"{loss_metrics['loss_mean']:.4f}"
                if "loss_mean" in loss_metrics
                else "n/a"
            )
            print(
                f"Step {step + 1:4d}/{effective_steps} | reward: {mean_reward:.3f}/3.000 | "
                f"exact: {reward_metrics['reward/exact/mean']:.1%} | "
                f"degenerate: {degenerate_fraction:.1%} | datums: {len(datums)} | "
                f"train_tokens: {train_tokens} | loss: {loss_text}"
            )

        print("Saving final LoRA weights for sampler...")
        final_weights = training_client.save_weights_for_sampler(name=run_name).result()
        print(f"Saved weights name: {run_name}, path: {final_weights.path}")
    except BaseException as error:
        if swanlab_run is not None:
            swanlab_run.finish(state="crashed", error=str(error))
        raise
    else:
        if swanlab_run is not None:
            swanlab_run.finish()


if __name__ == "__main__":
    cli_config = parse_args()
    started = time.perf_counter()
    main(cli_config)
    print("#" * 50)
    print("# all done")
    print(f"# train cost {time.perf_counter() - started:.2f}s")
    print("#" * 50)
