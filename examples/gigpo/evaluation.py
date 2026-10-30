"""独立测试入口：以相同任务、评分和预算评估基础策略或云端 LoRA 策略。

GiGPO/GRPO/PPO 的测试差异仅在待测 actor 权重，评估不计算训练优势或更新模型。
训练前不传 checkpoint；训练后必须传训练报告中的云端 actor checkpoint URI。
PPO 的 critic 只服务训练，评估时不创建 critic 或要求其 checkpoint。
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import replace
from numbers import Real

from triotune.agentrl import EpisodeResult
from triotune.api.client_api import call_async


def webshop_score_metrics(episodes: Sequence[EpisodeResult]) -> dict[str, float]:
    """从本轮全部 episode 的内存记录计算 WebShop 原始平均分和百分制 Score。

    episodes 包含每个目标的所有独立尝试，包括非法动作、预算退出和系统失败。
    只读取 score.components.task_score，不读取稀疏训练 reward 或过程奖励。
    没有购买的正常评分为零；系统失败没有 score 时同样计零，仍保留在分母。
    已有 score 却缺失 task_score，或分值不在有限的 [0,1] 内，直接报错，避免
    将评分契约错误伪装成低分。空评估也报错，不生成看似有效的零分结果。

    返回 task_score_mean（0–1）、score（均值乘 100）、task_score_count（有
    原始分值的尝试数）及 unscored_episodes（无评分、按零计入的尝试数）。
    指标以 eval/agent 为前缀；分母始终为 len(episodes)，每条尝试等权。
    """
    if not episodes:
        raise ValueError("WebShop Score requires at least one evaluation episode.")
    scores = []
    unscored = 0
    for episode in episodes:
        if episode.score is None:
            unscored += 1
            continue
        value = episode.score.components.get("task_score")
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                f"WebShop episode {episode.episode_id} requires a finite task_score in [0,1]."
            )
        scores.append(float(value))
    mean = math.fsum(scores) / len(episodes)
    return {
        "eval/agent/task_score_mean": mean,
        "eval/agent/score": mean * 100,
        "eval/agent/task_score_count": float(len(scores)),
        "eval/agent/unscored_episodes": float(unscored),
    }


async def load_actor_config(
    experiment, service, checkpoint: str | None, *, evaluation: bool
):
    """返回有效 PPOConfig 和可选的已加载 actor 客户端。

    experiment 是 train.Experiment；service 是 PyTRIO 服务客户端。checkpoint
    是 save_state 返回的远程 actor LoRA URI，不能用基础模型名替代。evaluation
    为真时关闭 critic，只保留 actor 的采样能力。加载失败直接报错，不退回基础
    模型；从训练状态导出的采样快照会包含基础权重和该云端 LoRA 的合成策略。
    """
    if checkpoint is not None and not checkpoint.strip():
        raise ValueError("checkpoint must be a nonempty cloud actor LoRA URI.")
    config = experiment.trainer
    if checkpoint and not evaluation:
        raise ValueError("Cloud checkpoint loading is evaluation-only.")
    if evaluation:
        config = replace(config, critic=replace(config.critic, enabled=False))
    client = None
    if checkpoint:
        client = await call_async(
            service, "create_training_client_from_state", path=checkpoint
        )
    return config, client


async def evaluate_policy(loop, *, checkpoint: str | None) -> dict:
    """用标准 AgentTrainingLoop.evaluate 测试，返回指标、权重来源和更新数。

    loop 已绑定数据、固定快照工厂和交互工作流。checkpoint 仅用于报告来源；
    实际加载已由 load_actor_config 完成。前后检查 actor/critic 更新计数，确保
    各算法测试都不执行训练。框架统计 success_rate、pass@1 和失败率；本模块
    从 last_eval_batch 的全部内存 episode 计算平均 task_score 与百分制 Score，
    并写入 trainer 指标和返回报告，不依赖可能不完整的磁盘 JSONL。
    """
    state = loop.trainer.state
    before = (state.optimizer_steps, state.critic_optimizer_steps)
    metrics = await loop.evaluate()
    score_metrics = webshop_score_metrics(loop.last_eval_batch.episodes)
    metrics.update(score_metrics)
    await loop.trainer.publish_metrics(score_metrics)
    after = (state.optimizer_steps, state.critic_optimizer_steps)
    if after != before:
        raise RuntimeError("Evaluation changed optimizer state.")
    return {
        "metrics": metrics,
        "optimizer_updates": after[0] - before[0],
        "critic_optimizer_updates": after[1] - before[1],
        "policy_source": "cloud_lora" if checkpoint else "base_model",
        "actor_lora_checkpoint": checkpoint,
    }


def main(argv=None):
    """接受与 train 共用的模型、算法、配置、划分和云端 checkpoint 参数。"""
    from .train import build_parser, run

    args = build_parser(evaluation=True).parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
