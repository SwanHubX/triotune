# 框架测试

测试的目标有两点

1. 某个算法框架可以通用于多个案例
2. 某个算法框架可以通用于多个算法

目前就完成第一项就行

## 测试方法

用pytrio设计案例，然后对比使用框架的结果，重点观察time、reward、kl等指标是否一致，或者误差很小

## 测试结果

1. grpo：[grpo框架测试结果（gms8k案例）](grpo_pytrio-use_eval.md)
2. dpo：[dpo框架测试结果](dpo_pytrio-use_eval.md)
3. opd：[opd框架测试结果（DeepMath 案例）](opd_pytrio-use_eval.md)
4. ppo：[PPO 算法族的测试方法、接口与 AReaL 功能边界](ppo_pytrio-use_eval.md)（本地测试及 actor/critic 双客户端两步远端训练已通过）
