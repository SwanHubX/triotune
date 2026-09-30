"""使用 ``trio_cookbook.trainer`` 训练 4x4 数独的 GRPO recipe。

这个脚本把 ``trio-cookbook/recipes/grpo_sudoku.py`` 的任务逻辑接到
PyTRIO cookbook 的 ``GRPOTrainer`` 上：数据格式、prompt、六项奖励和组内
advantage 都保持一致；rollout、Datum 对齐、远端 forward/backward、优化器
以及 checkpoint 则统一由 cookbook trainer 负责。

训练数据位于仓库 ``tests/sudoku_4x4_qa.jsonl``（答案中间文件为
``tests/sudoku_4x4_answer.jsonl``）。

最小运行：

    uv run python trio-cookbook/recipes/grpo_sudoku_trio.py \
        --steps 1 --batch-size 1 --group-size 4 --swanlab-mode disabled

    uv run python trio-cookbook/recipes/grpo_sudoku_trio.py \
        --steps 100 \
        --batch-size 2 \
        --group-size 6 \
        --max-tokens 1648 \
        --seed 42 \
        --swanlab-mode online

当前 cookbook 的 GRPOConfig 只支持 ``importance_sampling``；这也是
独立版 recipe 的默认训练方式。LoRA 权重会由 GRPOTrainer 按
``save-every`` 配置保存。
"""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import pytrio as trio
from datasets import Dataset, load_dataset
from tqdm import tqdm

from trio_cookbook.trainer import GRPOBatch, GRPOConfig as BaseGRPOConfig, GRPOResult, GRPOTrainer


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_FILE = REPOSITORY_ROOT / "tests" / "sudoku_4x4_qa.jsonl"
ANSWER_PATTERN = re.compile(r"^<think>[\s\S]*?</think>\s*<answer>([\s\S]*?)</answer>$")
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
COMPARISON_METRICS = {
    "reward/mean": "reward/mean",
    "reward/std": "reward/std",
    "reward/format/mean": "reward/format/mean",
    "reward/length/mean": "reward/length/mean",
    "reward/clue/mean": "reward/clue/mean",
    "reward/row/mean": "reward/row/mean",
    "reward/column/mean": "reward/column/mean",
    "reward/block/mean": "reward/block/mean",
    "reward/exact/mean": "reward/exact/mean",
    "policy/entropy": "policy/entropy",
    "policy/kl_sample_train": "policy/kl_sample_train",
    "rollout/completion_length_mean": "rollout/completion_length_mean",
    "rollout/completion_length_min": "rollout/completion_length_min",
    "rollout/completion_length_max": "rollout/completion_length_max",
    "rollout/degenerate_fraction": "rollout/degenerate_fraction",
    "trainer/loss": "train/loss",
    "trainer/tokens": "train/tokens",
    "optim/lr": "optim/lr",
    "optim/grad_norm": "optim/grad_norm",
}


@dataclass(slots=True)
class SudokuGRPOConfig(BaseGRPOConfig):
    """补齐独立版使用的 Adam 参数，使两个实验提交相同优化器配置。"""

    weight_decay: float = 0.0
    grad_clip_norm: float = 0.2

    def __post_init__(self) -> None:
        BaseGRPOConfig.__post_init__(self)
        if self.weight_decay < 0 or not math.isfinite(self.weight_decay):
            raise ValueError("weight_decay must be finite and non-negative")
        if self.grad_clip_norm < 0 or not math.isfinite(self.grad_clip_norm):
            raise ValueError("grad_clip_norm must be finite and non-negative")

    def optimizer_params(self) -> dict[str, float]:
        return {
            **BaseGRPOConfig.optimizer_params(self),
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
        }


@dataclass(frozen=True, slots=True)
class Arguments:
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
    save_every: int
    swanlab_mode: str
    swanlab_project: str
    run_name: str | None


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description="PyTRIO cookbook GRPO / 4x4 Sudoku")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--difficulty", choices=("simple", "medium", "difficult", "all"), default="simple")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--all-data", action="store_true", help="遍历过滤后的数据一次，忽略 --steps")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=1648)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.2)
    parser.add_argument("--save-every", type=int, default=0, help="每多少个 step 保存一次；0 表示不保存中间状态")
    parser.add_argument("--swanlab-mode", choices=("disabled", "online", "local", "offline"), default="online")
    parser.add_argument("--swanlab-project", default="sudoku-grpo-pytrio")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    for name in ("lora_rank", "steps", "batch_size", "group_size", "max_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.group_size < 2:
        parser.error("--group-size must be at least 2")
    if not 0 < args.top_p <= 1 or args.temperature <= 0:
        parser.error("--top-p must be in (0, 1] and --temperature must be positive")
    if args.save_every < 0:
        parser.error("--save-every must be non-negative")
    return Arguments(
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
        save_every=args.save_every,
        swanlab_mode=args.swanlab_mode,
        swanlab_project=args.swanlab_project,
        run_name=args.run_name,
    )


def format_question(question: str) -> str:
    return SUDOKU_FORMAT.format(*question)


def validate_row(row: dict[str, Any], index: int) -> None:
    question, answer = row.get("question"), row.get("answer")
    if not isinstance(question, str) or len(question) != 16 or any(c not in "1234_" for c in question):
        raise ValueError(f"row {index}: question must contain 16 characters from 1-4/_")
    if not isinstance(answer, str) or len(answer) != 16 or any(c not in "1234" for c in answer):
        raise ValueError(f"row {index}: answer must contain 16 digits from 1-4")
    if any(given != "_" and given != solved for given, solved in zip(question, answer, strict=True)):
        raise ValueError(f"row {index}: answer changes a given clue")


def load_sudoku_train(args: Arguments) -> Dataset:
    if not args.data_file.is_file():
        raise FileNotFoundError(
            f"Sudoku data not found at {args.data_file}; check tests/sudoku_4x4_qa.jsonl "
            "or pass --data-file."
        )
    dataset = load_dataset("json", data_files=str(args.data_file), split="train")
    if not isinstance(dataset, Dataset) or not {"question", "answer", "label"}.issubset(dataset.column_names):
        raise ValueError("Sudoku JSONL must contain question, answer, and label columns")
    if args.difficulty != "all":
        dataset = dataset.filter(lambda row: row["label"] == args.difficulty)
    if len(dataset) < 2:
        raise ValueError("at least two Sudoku rows are required after difficulty filtering")
    dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)["train"]
    for index, row in enumerate(dataset):
        validate_row(dict(row), index)

    def make_messages(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(format_question(row["question"]))}]
        }

    return dataset.map(make_messages)


def extract_board(response: str) -> list[int] | None:
    match = ANSWER_PATTERN.match(response)
    if match is None:
        return None
    numbers = re.findall(r"\d+", match.group(1))
    if len(numbers) != 16:
        return None
    return [int(number) for number in numbers]


def length_reward(response: str) -> float:
    length = len(response)
    if length < 1584:
        return 0.4
    return 0.4 * max(0.0, (1648 - length) / 64)


def uniqueness_score(groups: list[list[int]]) -> float:
    return sum(len(set(group)) / 4 for group in groups) / len(groups)


def parsed_scores(response: str, question: str, answer: str) -> dict[str, float]:
    """返回与独立版相同的格式、长度、线索、行、列、宫格奖励。"""
    board = extract_board(response)
    scores = {"format": 0.0, "length": length_reward(response), "clue": 0.0, "row": 0.0, "column": 0.0, "block": 0.0, "exact": 0.0}
    if board is None:
        return scores
    scores["format"] = 0.6 if all(1 <= value <= 4 for value in board) else 0.0
    given = [index for index, value in enumerate(question) if value != "_"]
    scores["clue"] = sum(board[index] == int(question[index]) for index in given) / len(given) if given else 0.0
    rows = [board[index : index + 4] for index in range(0, 16, 4)]
    columns = [[board[row * 4 + column] for row in range(4)] for column in range(4)]
    blocks = [
        [board[(block_row + row) * 4 + block_column + column] for row in range(2) for column in range(2)]
        for block_row in (0, 2)
        for block_column in (0, 2)
    ]
    scores["row"] = 0.4 * uniqueness_score(rows)
    scores["column"] = 0.3 * uniqueness_score(columns)
    scores["block"] = 0.3 * uniqueness_score(blocks)
    scores["exact"] = float(board == [int(value) for value in answer])
    return scores


def format_reward(completions: list[str], **_: Any) -> list[float]:
    return [parsed_scores(text, "________________", "1111111111111111")["format"] for text in completions]


def length_reward_func(completions: list[str], **_: Any) -> list[float]:
    return [length_reward(text) for text in completions]


def clue_reward(completions: list[str], question: list[str], **_: Any) -> list[float]:
    return [parsed_scores(text, puzzle, "1111111111111111")["clue"] for text, puzzle in zip(completions, question, strict=True)]


def row_reward(completions: list[str], **_: Any) -> list[float]:
    return [parsed_scores(text, "________________", "1111111111111111")["row"] for text in completions]


def column_reward(completions: list[str], **_: Any) -> list[float]:
    return [parsed_scores(text, "________________", "1111111111111111")["column"] for text in completions]


def block_reward(completions: list[str], **_: Any) -> list[float]:
    return [parsed_scores(text, "________________", "1111111111111111")["block"] for text in completions]


def exact_reward(completions: list[str], answer: list[str], **_: Any) -> list[float]:
    return [parsed_scores(text, "________________", target)["exact"] for text, target in zip(completions, answer, strict=True)]


for reward_function, reward_name in (
    (format_reward, "format"),
    (length_reward_func, "length"),
    (clue_reward, "clue"),
    (row_reward, "row"),
    (column_reward, "column"),
    (block_reward, "block"),
    (exact_reward, "exact"),
):
    reward_function.reward_name = reward_name


class SamplePrintingGRPOTrainer(GRPOTrainer):
    """在每个数独题目的 rollout 中打印第一个模型回答。"""

    def _groups(self, rows: Sequence[Mapping[str, Any]]) -> list[GRPOBatch]:
        """移除模板预填的 ``<think>``，要求模型生成完整标签和推理。"""

        groups = super()._groups(rows)
        raw_prefix = self._tokenizer.encode("<think>\n", add_special_tokens=False)
        if hasattr(raw_prefix, "ids"):
            raw_prefix = raw_prefix.ids
        if hasattr(raw_prefix, "tolist"):
            raw_prefix = raw_prefix.tolist()
        if not isinstance(raw_prefix, (list, tuple)) or not raw_prefix or not all(
            isinstance(token, Integral) and not isinstance(token, bool) for token in raw_prefix
        ):
            raise ValueError("tokenizer.encode must return the Qwen <think> prefix tokens")
        thinking_prefix = [int(token) for token in raw_prefix]
        for group in groups:
            if group.prompt_tokens[-len(thinking_prefix) :] != thinking_prefix:
                raise ValueError("Chat template must end with the Qwen <think> generation prefix")
            if not group.prompt.endswith("<think>\n"):
                raise ValueError("Rendered chat template must end with '<think>\\n'")
            group.prompt_tokens = group.prompt_tokens[: -len(thinking_prefix)]
            group.prompt = group.prompt[: -len("<think>\n")]
            group.sampling_prompt = trio.ModelInput.from_ints(group.prompt_tokens)
        return groups

    def _print_first_samples(self, groups: list[GRPOBatch]) -> None:
        for group in groups:
            if not group.samples:
                continue
            sample = group.samples[0]
            text = sample.text if sample.text is not None else self._decode(sample.tokens)
            question = group.row.get("question")
            title = format_question(question) if isinstance(question, str) else group.prompt_id
            tqdm.write(
                "\n"
                + "#" * 20
                + " 推理案例 "
                + "#" * 20
                + f"\n数独题目：\n{title}\n模型的第一个回答：\n{text}"
            )

    def _sample_groups(self, groups: list[GRPOBatch], client: Any) -> list[GRPOBatch]:
        sampled = super()._sample_groups(groups, client)
        self._print_first_samples(sampled)
        return sampled

    async def _sample_groups_async(
        self,
        groups: list[GRPOBatch],
        client: Any,
    ) -> list[GRPOBatch]:
        sampled = await super()._sample_groups_async(groups, client)
        self._print_first_samples(sampled)
        return sampled

    def _build_datums(
        self,
        groups: Sequence[GRPOBatch],
        reward_groups: Sequence[Sequence[float]],
    ) -> tuple[list[Any], int, dict[str, float]]:
        datums, skipped, metrics = super()._build_datums(groups, reward_groups)
        metrics["rollout/degenerate_fraction"] = skipped / len(groups)
        metrics["train/tokens"] = float(
            sum(
                float(value) != 0.0
                for datum in datums
                for value in datum.loss_fn_inputs["advantages"].data
            )
        )
        return datums, skipped, metrics

    def _log_swanlab(self, result: GRPOResult) -> None:
        if self._swanlab_run is None:
            return
        payload = {
            chart_name: float(result.metrics[source_name])
            for chart_name, source_name in COMPARISON_METRICS.items()
            if source_name in result.metrics
        }
        payload.update(
            {
                "trainer/step": float(result.step),
                "trainer/num_datums": float(result.num_datums),
                "trainer/skipped_groups": float(result.skipped_groups),
            }
        )
        self._swanlab_run.log(payload, step=result.step)


def make_config(args: Arguments, dataset_size: int) -> SudokuGRPOConfig:
    max_steps = math.ceil(dataset_size / args.batch_size) if args.all_data else args.steps
    return SudokuGRPOConfig(
        num_generations=args.group_size,
        max_completion_length=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        loss_fn="importance_sampling",
        reward_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        prompt_key="messages",
        chat_template_kwargs={"enable_thinking": True},
        seed=args.seed,
        save_every=args.save_every,
        mode=args.swanlab_mode,
        project=args.swanlab_project,
        run_name=args.run_name,
    )


def main(args: Arguments) -> None:
    dataset = load_sudoku_train(args)
    config = make_config(args, len(dataset))
    run_name = args.run_name or f"grpo-{args.base_model.rsplit('/', 1)[-1]}-sudoku"
    trainer = SamplePrintingGRPOTrainer(
        model_name=args.base_model,
        reward_funcs=[
            format_reward,
            length_reward_func,
            clue_reward,
            row_reward,
            column_reward,
            block_reward,
            exact_reward,
        ],
        args=config,
        train_dataset=dataset,
        lora_rank=args.lora_rank,
    )
    print(f"Loaded {len(dataset)} Sudoku examples; run={run_name}; steps={config.max_steps}")
    state = trainer.train()
    print(f"Training complete: step={state.global_step}, checkpoint={state.checkpoint_path}")


if __name__ == "__main__":
    main(parse_args())
