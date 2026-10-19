"""Run the huanhuan-chat SFT task directly with PyTRIO, without a trainer framework."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import time
from pathlib import Path

import pytrio as trio
import swanlab
from tqdm import tqdm


DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "huanhuan.json"
SYSTEM_PROMPT = "现在你要扮演皇帝身边的女人--甄嬛"


def datum(item: dict[str, str], tokenizer: object, max_length: int) -> tuple[trio.Datum, int]:
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["instruction"] + item["input"]},
    ]
    completion = {"role": "assistant", "content": item["output"]}
    prompt_ids = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )["input_ids"]
    full_ids = tokenizer.apply_chat_template(
        prompt + [completion], tokenize=True, add_generation_prompt=False, enable_thinking=False
    )["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("The tokenized prompt is not a prefix of the full example.")
    tokens = full_ids[:max_length]
    labels = [-100] * min(len(prompt_ids), len(tokens)) + tokens[len(prompt_ids) :]
    target = labels[1:]
    if not target or all(token == -100 for token in target):
        raise ValueError("The example has no completion tokens after truncation.")
    return (
        trio.Datum(
            model_input=trio.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={"target_tokens": target},
        ),
        sum(token != -100 for token in target),
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-in-flight", type=int, default=32)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    parser.add_argument("--swanlab", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--epochs, --batch-size and --limit must be positive")
    if args.max_in_flight < 1:
        parser.error("--max-in-flight must be positive")

    rows = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    if args.limit is not None:
        rows = rows[: args.limit]
    iterator = iter(itertools.chain.from_iterable(rows for _ in range(args.epochs)))
    client = await trio.ServiceClient().create_lora_training_client_async(
        base_model=args.model, rank=args.lora_rank, seed=args.seed
    )
    tokenizer = await asyncio.to_thread(client.get_tokenizer)
    optimizer = trio.AdamParams(
        learning_rate=args.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )
    run = swanlab.init(
        project="huanhuan-chat-sft",
        name=args.run_name or f"{args.model.rsplit('/', 1)[-1]}-sft-direct",
        config={
            "model": args.model, "batch_size": args.batch_size, "max_steps": args.steps,
            "max_in_flight": args.max_in_flight,
            "max_length": args.max_length, "learning_rate": args.learning_rate,
            "lora_rank": args.lora_rank, "seed": args.seed, "save_every": args.save_every,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ) if args.swanlab else None
    step = 0
    checkpoint = None
    metrics = {}
    progress = tqdm(total=args.steps, desc="SFT direct", unit="step")
    pending = []
    previous = None
    clock = [time.perf_counter()] * 2

    async def collect(forward_future, optim_future, token_count, index, started, saved, previous):
        nonlocal metrics, checkpoint
        forward = await forward_future
        optim = await optim_future
        latency = time.perf_counter() - started
        if previous is not None:
            await previous
        metrics = {
            str(key): float(value)
            for result in (forward, optim)
            for key, value in getattr(result, "metrics", {}).items()
            if isinstance(value, (int, float)) and math.isfinite(value)
        }
        if "loss_mean" in metrics:
            metrics["train/loss"] = metrics["loss_mean"]
        now = time.perf_counter()
        metrics.update({
            "train/tokens": float(token_count),
            "train/learning_rate": args.learning_rate,
            "time/request_latency": latency,
            "time/step": now - clock[1],
            "time/train": now - clock[0],
        })
        clock[1] = now
        if saved is not None:
            checkpoint = (await saved).path
        if run is not None:
            run.log({**metrics, "train/step": index}, step=index)
        progress.set_postfix({"loss": metrics.get("train/loss")}, refresh=False)
        progress.update(1)

    try:
        while args.steps is None or step < args.steps:
            if len(pending) >= args.max_in_flight:
                await pending.pop(0)
            while pending and pending[0].done():
                pending.pop(0).result()
            batch = list(itertools.islice(iterator, args.batch_size))
            if not batch:
                break
            started = time.perf_counter()
            prepared = [datum(item, tokenizer, args.max_length) for item in batch]
            forward_future = await client.forward_backward_async(
                [entry for entry, _ in prepared], loss_fn="cross_entropy", auto_shift=False
            )
            optim_future = await client.optim_step_async(optimizer)
            step += 1
            saved = None
            if args.save_every and step % args.save_every == 0:
                saved = await client.save_state_async(name=f"sft-step-{step:08d}")
            previous = asyncio.create_task(collect(
                forward_future, optim_future, sum(count for _, count in prepared),
                step, started, saved, previous,
            ))
            pending.append(previous)
        await asyncio.gather(*pending)
        if step and (not args.save_every or step % args.save_every):
            saved = await client.save_state_async(name=f"sft-step-{step:08d}")
            checkpoint = (await saved).path
        if run is not None:
            if checkpoint is not None:
                run.log_text(key="checkpoint/path", data=checkpoint, step=step)
            run.finish()
    except BaseException as error:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if run is not None:
            run.finish(state="crashed", error=str(error))
        raise
    finally:
        progress.close()
    print(f"steps={step} checkpoint={checkpoint} metrics={metrics}")


if __name__ == "__main__":
    asyncio.run(main())
