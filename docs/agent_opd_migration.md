# Agent OPD 迁移与验证

WebShop OPD 的通用功能已迁入根目录 `triotune/`。新增同步逐动作 OPD、独立
评估采样参数、工具后终止钩子、精简评估日志和各 trainer 共用的 sampler
权重保存。PPO 系列和 GiGPO 的优势、GAE、critic 与损失公式保持原实现。

## 兼容性

- 原根目录 manifest 缺少评估 `temperature/top_p` 时按 `1/1` 兼容；WebShop
  副本写入的默认字段也可加载。非默认评估参数改变仍会拒绝续训。
- 新 manifest 保留旧版 trainer state 字段集合，将 sampler URI、保存步骤和
  TTL 放入 `components.actor.sampler_checkpoint`。读取时恢复到
  `state.sampler_checkpoint_path`；也支持副本把 URI 存在 state 中的旧格式。
  旧格式没有保存步骤时不猜测，首次手动保存可能重新导出。
- 老版 loader 可以忽略 actor 扩展元数据，恢复原有默认配置的训练循环；
  它不能运行旧版不存在的 Agent OPD 或新增的非默认评估配置。
- 自定义 trainer 的 `create_sampling_snapshot(*, stop=None)` 继续支持默认
  训练和评估。使用非默认评估参数时，需要增加 `temperature/top_p` 参数。
  接口适配不会捕获并重试函数内部的 `TypeError`。
- 训练采集、训练更新和训练用 OpenAI backend 拒绝非 `1/1` 的采样快照。
  自定义 sampler 应继续满足原始 token 和未变换行为概率的契约。

WebShop 启动脚本优先使用根目录包，原副本保留供历史实验比对。修改
`sys.path` 不会替换已缓存的 Python 模块；请在新进程中启动入口，不要在同一
进程混用两份实现。可通过 `import triotune; print(triotune.__file__)` 确认来源。
入口会检查已加载的包；如果进程此前导入了副本，会明确报错要求重启，避免
继续使用错误版本。

## 续训与 Sampler 保留

训练恢复使用 `checkpoint.resume_from` 指向完整 JSON manifest，保留优化器、
数据游标、随机状态、采样 seed、评估和日志聚合窗口。Agent OPD 恢复时自动
延后循环绑定，直到 `AgentOPDTrainingLoop` 创建完成。

Sampler 只用于推理，配置和生命周期见 [包说明](../triotune/README.md)。
`ttl_seconds: null` 使用服务端默认保留策略；显式正整数请求对应的保留秒数，
不等于保证永久保存。导出只包含 student/actor，既不替换训练 checkpoint，
也不导出 critic。SFT 的流水线导出在对应 optimizer 之后立即提交。

定期保存和成功结束保存相互去重；中途暂停不触发结束保存。完整 manifest
记录已成功导出的步骤，恢复后保留去重状态。失败或取消不更新 sampler URI。
原有训练 manifest 使用原子替换，后续保存失败不会覆盖此前已发布的边界。

## 统一验证

在已有项目环境中安装可选测试依赖，不需要下载模型权重：

```bash
uv pip install --python .venv/bin/python pytest 'openai>=1.66,<3' 'fastapi>=0.100'
TRIOTUNE_TEST_TOKENIZER=Qwen/Qwen3.5-4B bash tests/scripts/test_training_stack.sh
```

`TRIOTUNE_TEST_TOKENIZER` 应指定本机已缓存的 tokenizer；没有缓存时可以省略，
相关集成测试会报告跳过。脚本设置 `HF_HUB_OFFLINE=1`，不会下载 tokenizer。
它依次执行根目录全量测试、额外 AgentRL 功能审计、WebShop 应用测试（目录
存在时），再构建 wheel 并从 wheel 导入模块，确认包内不含配置模板目录。
产物位于临时目录，或由 `TRIOTUNE_AUDIT_OUTPUT` 指定的目录。

配置默认值由 Python 配置类提供，运行时不读取包内 YAML 模板。用于配置
解析和往返验证的 YAML 位于 [tests/configs/checkpoint.yaml](../tests/configs/checkpoint.yaml)，
不随安装包发布；实验配置仍可放在应用自己的目录。

覆盖内容包括所有 PPO 算法预设、GiGPO、SFT/DPO/OPD、Agent OPD、OpenAI
官方 SDK/代理、多轮轨迹、奖励、异常/取消、原生 tokenizer、完整续训，以及
sampler 开关、周期、暂停、最终保存、actor/critic 隔离、权重对应和 TTL 传递。
CPU 传输夹具使用真实 Torch loss 和 Adam 状态，验证连续训练与分段续训的
权重、优化器、LR、输入和计数一致。

日志集成测试使用未 mock 的 SwanLab SDK 离线模式，对八类训练执行暂停和
续训，解析实际 `.swanlab` 数据记录，检查 loss、评估指标、sampler 文本路径
与 finish 记录；同时检查终端进度和损失打印、本地 JSONL 与 episode 证据。
与原有行为一致，新进程创建新的 SwanLab run，保留 global_step 和未完成的
指标聚合窗口。
单独保留这些日志进行检查可运行：

```bash
PYTHONPATH=.:tests SWANLAB_MODE=offline .venv/bin/python \
  tests/audit_training_reporting.py --output /tmp/triotune-reporting
```

离线日志可使用 SwanLab 的 `swanlab sync <run-directory>` 上传。上述测试不
声称已验证云端上传或真实远程模型训练，远程服务的实际权重保留期限仍由
服务端执行请求策略。
