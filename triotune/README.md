# PyTRIO 训练器


## 目录结构

```text
triotune/
├── api/              # 配置参数与 PyTRIO client 公共契约
├── dataset/          # 算法共享的数据校验、分词、mask 与 SWE 轨迹处理
├── trainer/          # 按算法组织的训练实现和稳定导入入口
│   ├── ppo/          # PPO/GRPO 等共享的采样、优势、损失与更新组件
│   ├── dpo/          # 偏好数据、成对损失与训练编排
│   ├── sft/          # 标签处理与异步流水线训练
│   └── opd/          # 教师评分、蒸馏信号与训练编排
└── utils/            # batch、Datum、指标与 checkpoint 跨模块实现
```

训练器模块的详细职责见 [trainer/README.md](trainer/README.md)。
