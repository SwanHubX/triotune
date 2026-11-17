# PyTRIO 训练器


## 目录结构

```text
triotune/
├── api/              # 配置参数与 PyTRIO client 公共契约
├── dataset/          # 算法共享的数据校验、分词与 mask
├── trainer/          # 按算法组织的训练实现和稳定导入入口
│   ├── ppo/          # PPO/GRPO 等共享的采样、优势、损失与更新组件
│   ├── dpo/          # 偏好数据、成对损失与训练编排
│   ├── sft/          # 标签处理与异步流水线训练
│   └── opd/          # 教师评分、蒸馏信号与训练编排
└── utils/            # batch、Datum、指标与 checkpoint 跨模块实现
```

训练器模块的详细职责见 [trainer/README.md](trainer/README.md)。

## Agent OPD 与 Sampler 权重保存

根目录包提供 `AgentOPDTrainingLoop`，复用 AgentRL 的多轮环境执行和
OPDTrainer 的 teacher 评分与优化。每轮原始动作单独训练；工具观察和此前
对话仅作上下文，环境奖励用于指标，不进入 vanilla OPD loss。

SFT、DPO、OPD、PPO 系列及 AgentRL 共用以下配置：

```yaml
checkpoint:
  save_every: 20
  state_dir: output/checkpoints
  sampler:
    enabled: true
    save_every: 50
    ttl_seconds: 2592000  # 请求保留 30 天
```

配置可直接写入实验 YAML，无需包内 `configs` 目录；默认值由
`CheckpointConfig` 和 `SamplerCheckpointConfig` 提供。`sampler.enabled`
默认关闭；`sampler.save_every: 0` 表示仅成功结束时保存，正数表示定期保存并
补存最终步骤。暂停不会触发结束保存，评估不会保存。SFT 在流水线中紧跟
对应 optimizer 提交导出，避免名称中的步骤与权重不一致。PPO 只导出 actor。

`save_state()` 保存训练权重和优化器；`save_weights_for_sampler()` 单独导出
可用于推理的权重，URI 位于 `state.sampler_checkpoint_path`，同时写入
SwanLab 的 `checkpoint/sampler_path` 和本地 JSONL。`ttl_seconds: null`
表示采用服务端保留策略，并不保证永久保存；需要固定保留期时请显式设置。
它不使用训练临时采样接口固定的 6 小时 TTL，也不会删除已有导出。

等待当前更新和 SFT 流水线完成后，可手动调用：

```python
path = await trainer.save_sampler_checkpoint_async()
# DPO 也支持同步调用：path = trainer.save_sampler_checkpoint()
```

手动保存也需要开启 sampler；相同步骤已成功导出时跳过。完整 checkpoint
记录 URI、导出步骤及请求的 TTL，重启后仍能去重。Sampler URI 不能代替包含
优化器、数据游标和 RNG 状态的训练 manifest。导出失败不会替换上一次 URI。

旧根目录和 WebShop 副本的 manifest 均可加载。默认评估参数的新增字段会在
循环身份比较时兼容处理；改变非默认评估配置仍拒绝续训。新 manifest 保持
旧版 trainer state 字段结构，将 sampler 元数据放入 actor 的扩展区域，
旧版 loader 可读取默认配置下的普通 trainer/AgentRL checkpoint。
旧版没有 Agent OPD 实现，无法运行其新增训练循环。

完整验证命令及覆盖范围见 [迁移验证说明](../docs/agent_opd_migration.md)。
