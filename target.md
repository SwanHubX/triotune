# Target

1. 明确 PyTRIO 的底层逻辑，查看使用哪些函数

```text
客户端 RL 框架                         PyTRIO / Trio-Train
──────────────────────────────────    ─────────────────────────────
任务、环境、prompt                     tokenizer / SamplingClient
rollout 编排、reward、advantage   ──▶  sample / compute_logprobs
token 对齐、Datum、loss 选择       ──▶  forward_backward
策略版本、更新顺序                 ──▶  optim_step
运行状态、评估、日志、恢复清单     ──▶  save_state / sampler weights
```

2. 设计前端使用 API
  1. 训练模块分块，尽量简化代码逻辑
  2. 后台逻辑参考 `tinker-cookbook/tinker-cookbook/rl/train.py` ，前台逻辑参考 `trl/trl/trainer/grpo_trainer.py`
  3. 测试是否能够正确运行


3. 同步异步改写，保持代码通用性
  1. 通用性初期保持为同样的算法可以应用于多种场合
  2. 后期可以改写为多种相似算法可以通用同一个前端API入口

4. 测试方法为：初期grpo多个项目（保存在recipes中）和通用框架改写不同任务的结果基本一致  


