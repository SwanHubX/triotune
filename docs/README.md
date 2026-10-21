# 框架测试

发布维护流程见 [发布 triotune 到 PyPI](releasing.md)。

测试的目标有两点

1. 某个算法框架可以通用于多个案例
2. 某个算法框架可以通用于多个算法

目前就完成第一项就行

## 测试方法

用pytrio设计案例，然后对比使用框架的结果，重点观察time、reward、kl等指标是否一致，或者误差很小

## 测试结果

1. dpo：[dpo框架测试结果](dpo_pytrio-use_eval.md)
2. opd：[opd框架测试结果（DeepMath 案例）](opd_pytrio-use_eval.md)
3. ppo：[PPO 算法族（含 GRPO）的测试方法、接口与 AReaL 功能边界](ppo_pytrio-use_eval.md)（本地测试及 actor/critic 双客户端两步远端训练已通过）

## 算法使用指南

- [PPO、GRPO 及相关算法：启动、接口与参数](algorithms/grpo_series_CN.md)
- [PPO, GRPO, and Related Algorithms: Usage and Configuration](algorithms/grpo_series.md)
