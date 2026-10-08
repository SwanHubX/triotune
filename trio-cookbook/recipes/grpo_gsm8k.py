"""PyTRIO 同步版 GRPO 教学脚本（只有 importance_sampling / ppo 两个内置 loss）。

GRPO 训练流程：
1. 从预处理好的 GSM8K JSONL 取一批数学题；
2. 用当前 LoRA 权重保存出采样客户端；
3. 每道题同步采样 group_size 个答案；
4. 用 boxed answer reward 计算 group-relative advantage；
5. 用 PyTRIO 内置 loss 做一次优化。

两个 loss 的区别（都是 policy gradient + importance sampling 修正）：

- importance_sampling：
      loss = -ratio * advantage
  其中 ratio = exp(当前策略 logprob - 采样时旧策略 logprob)。
  不做任何截断，新旧策略偏差大时训练容易不稳定。

- ppo：
      loss = -min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
  在 importance_sampling 基础上把 ratio 截断到 [1-eps, 1+eps]，
  防止一次更新把策略推得太远。eps 等参数由 PyTRIO 内置 ppo loss 管理。

最小试跑：

先运行 `bash trio-cookbook/scripts/train_grpo_gms8k.sh --prepare-only` 准备数据。

uv run python trio-cookbook/recipes/grpo_gsm8k.py \
    --steps 2 \
    --batch-size 2 \
    --group-size 4 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled

正式训练：

uv run python trio-cookbook/recipes/grpo_gsm8k.py \
    --steps 100 \
    --batch-size 4 \
    --group-size 8 \
    --max-tokens 1024 \
    --loss-fn importance_sampling \
    --swanlab-mode online \
    --workspace pytrio \
    --exp-name gsm8k_normal_pytrio
"""

import argparse
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm


LOSS_FNS = ("importance_sampling", "ppo")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_FILE = REPOSITORY_ROOT / "data" / "gsm8k_train.jsonl"


@dataclass
class GRPOConfig:
    """命令行参数解析后的训练配置。"""

    base_model: str
    data_file: Path
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
    loss_fn: str
    swanlab_mode: str
    swanlab_project: str
    workspace: str
    exp_name: str | None = None

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1 or self.max_tokens < 1 or self.lora_rank < 1:
            raise ValueError("steps, batch_size, max_tokens, and lora_rank must be positive")
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if self.loss_fn not in LOSS_FNS:
            raise ValueError(f"loss_fn must be one of {LOSS_FNS}")
        if not self.workspace.strip():
            raise ValueError("workspace must not be empty")
        if self.exp_name is not None and not self.exp_name.strip():
            raise ValueError("exp_name must not be empty")


@dataclass
class RolloutSample:
    """一条采样结果，以及构造 importance_sampling 所需的旧策略 logprobs。"""

    tokens: list[int]
    logprobs: list[float]
    text: str
    reward: float = 0.0
    advantage: float = 0.0


def parse_args() -> GRPOConfig:
    """把训练配置集中到命令行参数，避免依赖环境变量。"""
    parser = argparse.ArgumentParser(description="PyTRIO 同步版 GRPO / GSM8K 教学脚本")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B", help="PyTRIO 基础模型名")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE, help="已准备好的 GSM8K JSONL")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="GRPO 优化步数；每步从 GSM8K 取 batch-size 道题做 rollout",
    )
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="使用 GSM8K train split 全量数据训练一遍；打开后忽略 --steps",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="每个 step 的 GSM8K 题目数")
    parser.add_argument("--group-size", type=int, default=8, help="每道题采样的 completion 数")
    parser.add_argument("--max-tokens", type=int, default=1024, help="每次采样最多生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样 temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="采样 top_p")
    parser.add_argument("--seed", type=int, default=42, help="本地随机种子")
    parser.add_argument("--learning-rate", type=float, default=4e-5, help="Adam learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument(
        "--loss-fn",
        choices=LOSS_FNS,
        default="importance_sampling",
        help="训练 loss：importance_sampling / ppo",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
        default="online",
        help="SwanLab 记录模式",
    )
    parser.add_argument("--swanlab-project", default="cookbook-grpo-test", help="SwanLab project")
    parser.add_argument("--workspace", default="pytrio", help="SwanLab 组织或工作空间")
    parser.add_argument("--exp-name", help="SwanLab 实验名称；默认按训练参数自动生成")
    args = parser.parse_args()

    return GRPOConfig(
        base_model=args.base_model,
        data_file=args.data_file,
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
        loss_fn=args.loss_fn,
        swanlab_mode=args.swanlab_mode,
        swanlab_project=args.swanlab_project,
        workspace=args.workspace,
        exp_name=args.exp_name,
    )


def extract_boxed(text: str) -> str | None:
    """取最后一个 \\boxed{...} 作为模型最终答案。"""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(text: str) -> str:
    """GSM8K 答案只做轻量归一化，避免 1,000 和 1000 被判成不同。"""
    return text.replace(",", "").strip().rstrip(".")


def grade_answer(response: str, ground_truth: str) -> float:
    """boxed answer 与标准答案完全一致时给 1，否则给 0。"""
    answer = extract_boxed(response)
    if answer is None:
        return 0.0
    return 1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0


def extract_gsm8k_answer(answer_text: str) -> str:
    """接受原始 GSM8K 答案或 JSONL 中预处理后的数值答案。"""
    match = re.search(r"####\s*(.+)", answer_text)
    answer = normalize_answer(match.group(1) if match else answer_text)
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", answer):
        raise ValueError(f"Invalid GSM8K final answer: {answer_text!r}")
    return answer


def build_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    """按框架相同的 chat template 设置渲染已准备好的 prompt。"""
    prompt_tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(prompt_tokens, Mapping):
        prompt_tokens = prompt_tokens.get("input_ids")
    elif hasattr(prompt_tokens, "input_ids"):
        prompt_tokens = prompt_tokens.input_ids
    if hasattr(prompt_tokens, "tolist"):
        prompt_tokens = prompt_tokens.tolist()
    if (
        isinstance(prompt_tokens, (list, tuple))
        and len(prompt_tokens) == 1
        and isinstance(prompt_tokens[0], (list, tuple))
    ):
        prompt_tokens = prompt_tokens[0]
    if not isinstance(prompt_tokens, (list, tuple)) or not prompt_tokens or not all(
        isinstance(token, Integral) and not isinstance(token, bool) for token in prompt_tokens
    ):
        raise ValueError("Chat template must return a non-empty integer token sequence")
    return [int(token) for token in prompt_tokens]


def load_gsm8k_train(data_file: Path) -> Dataset:
    """读取由独立数据准备脚本生成的 JSONL，不在训练入口中构建数据。"""
    path = data_file.expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"GSM8K data not found at {path}; prepare it with "
            "bash trio-cookbook/scripts/train_grpo_gms8k.sh --prepare-only"
        )
    dataset = load_dataset("json", data_files=str(path), split="train")
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset)!r}")
    if not {"prompt", "answer", "question"}.issubset(dataset.column_names):
        raise ValueError("GSM8K JSONL must contain prompt, answer, and question columns")
    if not dataset:
        raise ValueError(f"No GSM8K rows found in {path}")
    for index, row in enumerate(dataset):
        messages = row["prompt"]
        if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise ValueError(f"Invalid chat prompt in GSM8K row {index}")
        if not isinstance(row["answer"], str):
            raise ValueError(f"Invalid answer in GSM8K row {index}")
        extract_gsm8k_answer(row["answer"])
    return dataset


def run_rollout_group(
    sampling_client: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    sampling_params: trio.SamplingParams,
    group_size: int,
) -> list[RolloutSample]:
    """同步采样一个 prompt 的 group_size 个 completion。"""
    # 对同一个 prompt 一次采样 group_size 个回答；同步 API 返回 future，所以这里立刻 `.result()` 等待。
    result = sampling_client.sample(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=group_size,
        sampling_params=sampling_params,
        return_text=True,
    ).result()

    if len(result.sequences) != group_size:
        raise ValueError(f"Expected {group_size} samples, got {len(result.sequences)}")
    samples: list[RolloutSample] = []
    for sequence in result.sequences:
        # return_text=True 时通常会直接返回文本；如果没有文本，就用 tokenizer 从 token 解码。
        text = sequence.text
        if text is None:
            text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)

        # PyTRIO 采样返回 completion token 的 logprobs；
        # 后续 importance_sampling / ppo loss 需要用这组 logprobs 作为“采样时旧策略”的概率。
        tokens = list(sequence.tokens)
        logprobs = [float(value) for value in sequence.logprobs]
        if len(tokens) != len(logprobs):
            raise ValueError(
                f"Generated token/logprob length mismatch: {len(tokens)} != {len(logprobs)}"
            )

        samples.append(RolloutSample(tokens=tokens, logprobs=logprobs, text=text))
    return samples


def score_rollout_group(samples: list[RolloutSample], ground_truth: str) -> None:
    """计算与框架相同的 boxed reward 和组内中心化 advantage。"""
    rewards = [grade_answer(sample.text, ground_truth) for sample in samples]
    mean_reward = sum(rewards) / len(rewards)
    for sample, reward in zip(samples, rewards, strict=True):
        sample.reward = reward
        sample.advantage = reward - mean_reward


def build_grpo_datum(prompt_tokens: list[int], sample: RolloutSample) -> trio.Datum:
    """把单条 completion 转成 PyTRIO importance_sampling / ppo 所需的 Datum。"""
    if not sample.tokens:
        raise ValueError("Cannot train on an empty completion")

    # 自回归对齐方式如下：
    # input = prompt + completion[:-1]
    # target 前 observation_len 个位置属于 prompt 内部预测，不训练，用 0 / 0.0 占位；
    # 从最后一个 prompt token 开始预测 completion 的每个 token。
    observation_len = len(prompt_tokens) - 1
    input_tokens = prompt_tokens + sample.tokens[:-1]
    target_tokens = [0] * observation_len + sample.tokens
    padded_logprobs = [0.0] * observation_len + sample.logprobs
    padded_advantages = [0.0] * observation_len + [sample.advantage] * len(sample.tokens)

    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(padded_logprobs)
        == len(padded_advantages)
    ):
        raise ValueError("GRPO datum fields must have the same token length")

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
            "advantages": np.asarray(padded_advantages, dtype=np.float32),
        },
    )


def run_optimization_step(
    training_client: Any,
    datums: list[trio.Datum],
    *,
    loss_fn: str,
    adam_params: trio.AdamParams,
) -> dict[str, Any]:
    """Run one failure-safe forward/backward and optimizer boundary."""

    if not datums:
        return {}
    forward_future = training_client.forward_backward(datums, loss_fn=loss_fn)
    # Do not submit optim_step until forward/backward has completed
    # successfully; otherwise a failed forward could apply stale gradients.
    forward_result = forward_future.result()
    optimizer_future = training_client.optim_step(adam_params)
    optimizer_result = optimizer_future.result()
    metrics = dict(forward_result.metrics)
    optimizer_metrics = getattr(optimizer_result, "metrics", None)
    if isinstance(optimizer_metrics, Mapping):
        metrics.update(optimizer_metrics)
    kl = sample_train_kl(datums, forward_result)
    if kl is not None:
        metrics["policy/kl_sample_train"] = kl
    return metrics


def sample_train_kl(datums: list[trio.Datum], forward_result: Any) -> float | None:
    """和 GRPOTrainer 一样，仅在非零 advantage token 上计算采样/训练 KL。"""
    outputs = getattr(forward_result, "loss_fn_outputs", None)
    if not isinstance(outputs, list) or len(outputs) != len(datums):
        return None

    differences: list[float] = []
    for datum, output in zip(datums, outputs, strict=True):
        if not isinstance(output, Mapping) or "logprobs" not in output:
            return None
        inputs = getattr(datum, "loss_fn_inputs", None)
        if not isinstance(inputs, Mapping):
            return None
        tensors = [
            getattr(inputs.get("logprobs"), "data", None),
            getattr(inputs.get("advantages"), "data", None),
            getattr(output["logprobs"], "data", None),
        ]
        if not all(isinstance(values, list) for values in tensors):
            return None
        old_logprobs, advantages, current_logprobs = tensors
        if not len(old_logprobs) == len(advantages) == len(current_logprobs):
            return None
        differences.extend(
            (float(old) - float(current)) ** 2
            for old, advantage, current in zip(
                old_logprobs, advantages, current_logprobs, strict=True
            )
            if float(advantage) != 0.0
        )
    return 0.5 * sum(differences) / len(differences) if differences else None


def get_num_steps(dataset: Dataset, config: GRPOConfig) -> int:
    """计算实际训练 step 数；all-data 模式会覆盖命令行里的 steps。"""
    if config.all_data:
        return (len(dataset) + config.batch_size - 1) // config.batch_size
    return config.steps


def model_slug(base_model: str) -> str:
    """把模型名压成适合 experiment / weights name 的短名称。"""
    name = base_model.rsplit("/", 1)[-1].lower()
    name = name.replace("qwen3.5", "qwen35")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def build_run_name(config: GRPOConfig, effective_steps: int) -> str:
    """SwanLab experiment name 和最终权重名共用同一个规则生成值。"""
    loss_slug = config.loss_fn.replace("_", "-")
    return (
        f"grpo-sync-{model_slug(config.base_model)}-gsm8k-"
        f"{loss_slug}-steps{effective_steps}"
    )


def pick_batch(dataset: Dataset, step: int, batch_size: int, all_data: bool) -> Dataset:
    """取当前 step 的 batch；all-data 模式下不回绕，确保每条样本最多用一次。"""
    start = step * batch_size
    if all_data:
        end = min(start + batch_size, len(dataset))
        indices = list(range(start, end))
    else:
        # 非 all-data 模式保留原来的回绕逻辑，允许 steps 超过数据集可切出的完整 batch 数。
        indices = [(start + offset) % len(dataset) for offset in range(batch_size)]
    return dataset.select(indices)


def init_swanlab_run(
    config: GRPOConfig,
    effective_steps: int,
    dataset_size: int,
    run_name: str,
) -> Any | None:
    """SwanLab 只记录关键 GRPO 指标，不影响主训练逻辑。"""
    if config.swanlab_mode == "disabled":
        return None
    return swanlab.init(
        project=config.swanlab_project,
        workspace=config.workspace,
        name=config.exp_name or run_name,
        mode=config.swanlab_mode,
        config={
            "base_model": config.base_model,
            "data_file": str(config.data_file),
            "run_name": run_name,
            "exp_name": config.exp_name or run_name,
            "lora_rank": config.lora_rank,
            "steps": config.steps,
            "all_data": config.all_data,
            "effective_steps": effective_steps,
            "batch_size": config.batch_size,
            "dataset_size": dataset_size,
            "group_size": config.group_size,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "learning_rate": config.learning_rate,
            "beta1": config.beta1,
            "beta2": config.beta2,
            "loss_fn": config.loss_fn,
            "swanlab_mode": config.swanlab_mode,
            "seed": config.seed,
            "weights_name": run_name,
        },
    )


def get_numeric_metric(loss_metrics: dict[str, Any], key: str) -> float | None:
    """按当前 PyTRIO 实际返回的 metric key 精确取值，不做模糊匹配。"""
    value = loss_metrics.get(key)
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None


def count_train_tokens(datums: list[trio.Datum]) -> int:
    """统计非零 advantage 的 token 数，也就是实际参与 RL loss 的 completion token。"""
    total = 0
    for datum in datums:
        total += sum(
            1
            for value in datum.loss_fn_inputs["advantages"].data
            if float(value) != 0.0
        )
    return total


def build_step_metrics(
    groups: list[list[RolloutSample]],
    datums: list[trio.Datum],
    skipped_groups: int,
    training_metrics: dict[str, Any],
    learning_rate: float,
) -> dict[str, float]:
    """用与 GRPOTrainer 相同的名称和分母汇总一步训练指标。"""
    samples = [sample for group in groups for sample in group]
    rewards = [sample.reward for sample in samples]
    lengths = [len(sample.tokens) for sample in samples]
    logprobs = [value for sample in samples for value in sample.logprobs]
    reward_mean = sum(rewards) / len(rewards)
    reward_std = math.sqrt(sum((value - reward_mean) ** 2 for value in rewards) / len(rewards))
    length_mean = sum(lengths) / len(lengths)

    metrics = {
        "reward/mean": reward_mean,
        "reward/std": reward_std,
        "reward/boxed_answer_reward/mean": reward_mean,
        "rollout/completion_length_mean": length_mean,
        "rollout/completion_length_min": float(min(lengths)),
        "rollout/completion_length_max": float(max(lengths)),
        "optim/lr": learning_rate,
        "trainer/num_datums": float(len(datums)),
        "trainer/skipped_groups": float(skipped_groups),
        # Preserve the teaching recipe's original dashboard keys.
        "reward": reward_mean,
        "frac_degenerate": skipped_groups / len(groups),
        "rollout/avg_gen_len": length_mean,
        "datums": float(len(datums)),
        "train_tokens": float(count_train_tokens(datums)),
    }
    if logprobs:
        metrics["policy/entropy"] = -sum(logprobs) / len(logprobs)
    if not datums:
        metrics["trainer/skipped"] = 1.0

    for key, value in training_metrics.items():
        if isinstance(value, int | float) and math.isfinite(float(value)):
            if "/" in key:
                metrics[key] = float(value)
            else:
                metrics[f"trainer/{key}"] = float(value)
    for source in ("loss_mean", "loss"):
        if (loss := get_numeric_metric(training_metrics, source)) is not None:
            metrics["trainer/loss"] = loss
            metrics["loss"] = loss
            metrics["loss_mean"] = loss
            break
    for source in ("optim/grad_norm", "grad_norm", "gradient_norm"):
        if (grad_norm := get_numeric_metric(training_metrics, source)) is not None:
            metrics["optim/grad_norm"] = grad_norm
            break
    return metrics


def main(config: GRPOConfig) -> None:
    run_started = time.perf_counter()
    np.random.seed(config.seed)

    print(f"Loading GSM8K dataset from {config.data_file}...")
    train_data = load_gsm8k_train(config.data_file)
    print(f"Loaded {len(train_data)} GSM8K training examples")
    effective_steps = get_num_steps(train_data, config)
    run_name = build_run_name(config, effective_steps)
    print(f"Run / weights name: {run_name}")
    if config.all_data:
        print(
            f"All-data mode: {effective_steps} steps will cover "
            f"{len(train_data)} examples once"
        )

    print("Creating PyTRIO clients...")
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
    )
    adam_params = trio.AdamParams(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=1e-8,
    )
    swanlab_run = init_swanlab_run(
        config=config,
        effective_steps=effective_steps,
        dataset_size=len(train_data),
        run_name=run_name,
    )

    try:
        for step in range(effective_steps):
            step_started = time.perf_counter()
            batch_rows = pick_batch(
                train_data,
                step,
                config.batch_size,
                config.all_data,
            )
            prepared = [
                (build_prompt(tokenizer, row["prompt"]), extract_gsm8k_answer(row["answer"]))
                for row in batch_rows
            ]

            rollout_started = time.perf_counter()
            # 每步发布当前 LoRA 权重，和 GRPOTrainer 一样使用当前策略采样。
            sampling_client = training_client.save_weights_and_get_sampling_client()
            groups = [
                run_rollout_group(
                    sampling_client=sampling_client,
                    tokenizer=tokenizer,
                    prompt_tokens=prompt_tokens,
                    sampling_params=sampling_params,
                    group_size=config.group_size,
                )
                for prompt_tokens, _ in tqdm(prepared, desc=f"GRPO step {step + 1}", unit="prompt")
            ]
            rollout_time = time.perf_counter() - rollout_started

            reward_started = time.perf_counter()
            for samples, (_, ground_truth) in zip(groups, prepared, strict=True):
                score_rollout_group(samples, ground_truth)
            reward_time = time.perf_counter() - reward_started

            train_started = time.perf_counter()
            datums: list[trio.Datum] = []
            skipped_groups = 0
            for samples, (prompt_tokens, _) in zip(groups, prepared, strict=True):
                if all(sample.advantage == 0.0 for sample in samples):
                    skipped_groups += 1
                    continue
                for sample in samples:
                    datums.append(build_grpo_datum(prompt_tokens, sample))

            # importance_sampling / ppo 都是 PyTRIO 内置 loss，datum 只需要
            # target_tokens、采样时旧策略 logprobs 和 GRPO 组内 advantage。
            training_metrics = run_optimization_step(
                training_client,
                datums,
                loss_fn=config.loss_fn,
                adam_params=adam_params,
            )
            log_payload = build_step_metrics(
                groups, datums, skipped_groups, training_metrics, config.learning_rate
            )
            train_time = time.perf_counter() - train_started
            log_payload.update(
                {
                    "trainer/step": float(step + 1),
                    "time/rollout": rollout_time,
                    "time/reward": reward_time,
                    "time/train": train_time,
                    "time/step": time.perf_counter() - step_started,
                }
            )
            if swanlab_run is not None:
                swanlab_run.log(
                    {
                        (
                            "train/learning_rate"
                            if key == "optim/lr"
                            else f"train/{key.partition('/')[2]}"
                            if key.startswith(("trainer/", "scalar/"))
                            else f"train/{key}" if "/" not in key else key
                        ): value
                        for key, value in log_payload.items()
                    },
                    step=step + 1,
                )

            loss_mean = log_payload.get("trainer/loss")
            loss_mean_text = "n/a" if loss_mean is None else f"{loss_mean:.4f}"
            kl = log_payload.get("policy/kl_sample_train")
            kl_text = "n/a" if kl is None else f"{kl:.4f}"
            print(
                f"Step {step + 1:2d} | reward: {log_payload['reward/mean']:.3f} | "
                f"degenerate: {log_payload['frac_degenerate']:.0%} | datums: {len(datums)} | "
                f"loss: {loss_mean_text} | kl: {kl_text} | "
                f"time: {log_payload['time/step']:.2f}s | loss_fn: {config.loss_fn}"
            )

        print("Saving final LoRA weights for sampler...")
        final_weights = training_client.save_weights_for_sampler(
            name=run_name
        ).result()
        print(f"Saved weights name: {run_name}, path: {final_weights.path}")
        run_time = time.perf_counter() - run_started
        print(f"Training finished in {run_time:.2f}s")
        if swanlab_run is not None:
            swanlab_run.log({"time/run": run_time}, step=effective_steps)
            swanlab_run.log_text(key="weights/path", data=final_weights.path, step=effective_steps)
    except BaseException as error:
        if swanlab_run is not None:
            swanlab_run.finish(state="crashed", error=str(error))
        raise
    else:
        if swanlab_run is not None:
            swanlab_run.finish()


if __name__ == "__main__":
    main(parse_args())
