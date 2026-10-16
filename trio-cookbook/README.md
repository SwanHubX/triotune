# PyTRIO 训练器

本目录提供基于 PyTRIO 客户端的纯文本 GRPO、DPO、SFT、OPD 和 PPO-family 训练器及运行示例。


## 目录结构

```text
trio-cookbook/
├── api/              # 配置参数与 PyTRIO client 公共契约
├── configs/          # GRPO、DPO、SFT、OPD 训练参数
├── dataset/          # 算法共享的数据校验、分词、mask 与 SWE 轨迹处理
├── recipes/          # 框架版和独立版训练入口
├── scripts/          # 启动脚本及使用说明
├── trainer/          # 按算法组织的训练实现和稳定导入入口
│   ├── grpo/         # 分组采样、奖励、优势、Datum 与训练编排
│   ├── dpo/          # 偏好数据、成对损失与训练编排
│   ├── sft/          # 标签处理与异步流水线训练
│   └── opd/          # 教师评分、蒸馏信号与训练编排
└── utils/            # batch、Datum、指标与 checkpoint 跨模块实现
```

训练器模块的详细职责见 [trainer/README.md](trainer/README.md)；脚本参数及独立版示例见
[scripts/README.md](scripts/README.md)。
