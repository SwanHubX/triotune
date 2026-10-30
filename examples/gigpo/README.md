# WebShop 中的 GiGPO 案例研究

本案例研究 GiGPO 在多步购物任务中的信用分配，并提供 GRPO、PPO 对照和状态级优势消融接口。交互、轨迹导出、优化和评估复用 TrioTune 已有实现；商品检索、页面跳转、目标生成和购物评分直接使用 [Princeton 原始 WebShop](https://github.com/princeton-nlp/WebShop) 的文本模拟器。当前提供可运行的实验流程，不包含已经完成的论文性能复现结论。

## 算法原理与代码对应

同一购物目标采样多个独立候选。GiGPO 在目标内部比较完整 episode 的回报，再在同一目标内部，把到达相同锚点状态的决策组成更小的组，比较它们的后续回报。每个决策获得联合优势：

$$
A_{i,t} = A_i^{episode} + \omega A_{i,t}^{state}
$$

默认使用组内减均值。episode 项比较总环境回报；state 项比较从当前决策开始、按回合折扣的 return-to-go。只有一个决策的状态组没有可比较动作，state 项为零。可通过 gigpo.mode 切换为同时除以样本标准差的方式，通过 gigpo.discount 设置回合折扣。

| 算法步骤 | 对应代码 | 作用 |
| --- | --- | --- |
| 同目标候选分组 | [data_prepare.py](data_prepare.py) 的 task_rows；标准 AgentRolloutWorkflow.collect | 任务行保留目标身份，框架为每个目标扩展多个候选 |
| 候选环境隔离 | [environment.py](environment.py) 的 SessionPool、ShoppingEpisodeExecutor | 每条 rollout 独立 browser/session，商品与检索索引共享 |
| 锚点状态锁定 | [environment.py](environment.py) 的 WebShopAnchors.observe；[标准 EpisodeRunner](../../triotune/agentrl/runtime/runner.py) | 动作生成前读取页面快照，按 interaction_id 保存 |
| episode 与 state 优势 | [GiGPOAdvantageProvider](../../triotune/agentrl/training/gigpo.py) | 同目标内按结构化状态精确分组，计算并合并两级优势 |
| 优势与原始 token 对齐 | [标准轨迹导出器](../../triotune/agentrl/rollout/trajectory.py) | 保留原始 token/logprob、交互 ID 和 loss mask，页面及工具 token 不参与策略损失 |
| 优化与策略刷新 | [train.py](train.py)；[AgentTrainingLoop](../../triotune/agentrl/training/loop.py) | 完整批采样后调用标准 PPOTrainer.step，再导出新策略快照 |

### 锚点状态如何锁定

锁定顺序是：打开独立会话 → 读取当前页面、可用动作及 done → 深拷贝状态 → runner 按 interaction_id 再保存一份快照 → 生成动作 → 执行工具 → 得到下一页面。后续搜索或点击不会修改前一步已经保存的锚点。

锚点只包含模型当时可见的环境状态，排除 rollout ID、隐藏商品答案和完整对话历史。不同候选在同一目标下看到完全相同的结构化页面时才进入同一状态组；不同目标不会合组。模型接收购物目标和全部历史交互；锚点只用于算法分组，不决定模型上下文。

这里锁定的是动作前状态的记录及比较身份，不会强迫候选执行相同动作或停留在同一页面。候选采取不同动作后的后续回报，才是 state 项比较的对象。整个更新批使用同一策略快照，禁止异步采样和上下文压缩。每次请求沿用此前全部交互及其原始 token，不摘要、截断或改写已经采样的动作。

## 模块与数据流

| 文件 | 具体职责 |
| --- | --- |
| [data_prepare.py](data_prepare.py) | HF 缓存定位、资产指纹、数据链接、1k Lucene 索引、目标索引划分和任务行；也提供数据准备 CLI |
| [environment.py](environment.py) | 模拟器加载、独立购物会话、交互适配器组装、可信终局与错误评分、过程奖励研究钩子、动作前锚点、资源清理 |
| [train.py](train.py) | 路径解析与配置校验，连接 PPOConfig、AgentConfig、GiGPOConfig；环境自检、采样探测、训练、定期评估和统一启动入口 |
| [evaluation.py](evaluation.py) | 各算法训练前/训练后的独立测试、云端 actor LoRA checkpoint 加载、禁止优化检查；训练入口的 checkpoint 评估也调用此模块 |
| [configs/grpo-webshop-configs.yaml](configs/grpo-webshop-configs.yaml) | 正式 GRPO 对照配置 |
| [configs/gigpo-webshop-configs.yaml](configs/gigpo-webshop-configs.yaml) | 从头训练的原始 GiGPO 配置 |
| [configs/gigpo-webshop-resume-configs.yaml](configs/gigpo-webshop-resume-configs.yaml) | 从旧 step-50 checkpoint 续训到 step 100 |
| [scripts/run-grpo-webshop.sh](scripts/run-grpo-webshop.sh) | 从任意工作目录启动 GRPO 训练 |
| [scripts/run-gigpo-webshop.sh](scripts/run-gigpo-webshop.sh) | 从任意工作目录启动 GiGPO 训练 |
| [scripts/run-gigpo-webshop-resume.sh](scripts/run-gigpo-webshop-resume.sh) | 启动旧 checkpoint 兼容续训 |
| [requirements-webshop.txt](requirements-webshop.txt) | 模拟器额外依赖，不修改项目依赖锁文件 |

```text
HF 缓存中的商品与属性
  → 固定源码版本和环境 seed → 上游 WebShop 生成固定顺序的 goals
  → 按协议划分 goal_index → 每个训练目标扩展多个独立购物会话
  → 固定策略快照 → 每次动作前锁定可见页面锚点
  → 标准 EpisodeRunner 保存原始 token/logprob、页面、动作与可信奖励
  → 标准轨迹导出器保留交互身份与 loss mask
  → GiGPOAdvantageProvider 计算 episode + state 联合优势
  → 标准 PPOTrainer.step 更新 → 新策略快照 → 定期评估及云端 checkpoint
  → evaluation.py 使用相同任务协议测试基础模型或选定的云端 LoRA 策略
```

数据固定为 YWZBrandon/webshop-data 的 ce990fff5aee388db2706f07820c578ab68e0453 版本。商品来自 items_shuffle_1000.json，属性来自 items_ins_v2_1000.json。旧 loader 还无条件读取 items_human_ins.json，因此准备时也链接它，但环境设置 human_goals=False，不使用人工指令生成目标。

1,000 个商品构成共享目录，任务由商品属性、选项和价格约束生成，目标数量可能大于商品数量。任务行只保存 task_id 和 goal_index；初始指令及页面由 reset 产生，环境结果由 step 产生，无需占位 parquet 文件。

### 划分、奖励和实验边界

environment.seed 固定商品价格与目标价格约束，上游 SimServer 随后使用其原生固定种子 233 打乱 goals。索引指向这份可复现的目标列表，而非商品编号。以下区间均为左闭右开，N 为实际目标数。

| data.split_protocol | 梯度更新 | 训练中的监控评估 | 显式最终测试 |
| --- | --- | --- | --- |
| official，默认 | [1500,N) | [500,1500) | [0,500)，训练期间不运行 |

默认使用 official 三段索引划分，训练过程中只监控 eval，选择 checkpoint 后再显式运行 test。为兼容已有实验，代码仍接受 verl 索引协议，它只改变目标划分，不引入其他仓库或训练框架。两种协议都共享商品目录，衡量目标级任务表现；1k 商品和合成目标的结果也不等于完整 WebShop 人工任务基准。切换环境来源会改变目标内容与顺序，已有分支版本的 goal_index 和指标不能直接沿用，应重新训练并核对 goal 指纹。

默认 reward.mode=sparse：完整结束且购买原始 task_score=1 的商品才奖励 10，其余为零。dense 使用购买后的 [0,1] 原始部分匹配分数，仍属于终局奖励。原始 task_score 始终保留在评分 components 中。

为兼容指定的旧 checkpoint，案例恢复纯 JSON 单动作协议并设置 enable_thinking=false。每回合只允许一个 search 或 click；购买后模型还需输出 `{"final":"done"}` 才算完整结束。非法 JSON 结束当前 episode，合法但不可执行的工具动作返回错误观察。没有默认非法动作负奖励，也没有购买后自动终止。

concat 模式把一条完整 episode 导出为一个训练行，保留全部历史上下文。历史页面和模板 token 作为 masked context，模型生成 token 参与策略损失；没有滑动窗口或上下文压缩。

rollout/completion_length_mean 恢复为完整轨迹训练行的 completion 长度，其中包含轮次之间的工具观察；它仍不等于纯模型输出量。比较实际模型输出长度应看 agent/generated_tokens_mean，联合 agent/turns_mean、agent/valid_tool_calls_mean 和 reward/invalid_action_count/mean 判断交互是否正常。

## 对照与消融

| 实验 | 配置或接口 | 保留和变化的内容 |
| --- | --- | --- |
| GiGPO | algorithm=gigpo | 无 critic，插件计算完整联合优势，禁止二次归一化 |
| GRPO | algorithm=grpo | 无插件、无 critic，使用框架 GRPO 组奖励归一化及优势路径 |
| PPO | algorithm=ppo | 无插件，启用独立云端 critic，使用框架 GAE |
| 移除状态级信用分配 | GiGPO 下设置 step_advantage_weight=0 | 保留 episode 相对优势，环境奖励和采样保持一致 |
| 过程奖励开关 | environment.build_workflow 的 process_reward_fn | 默认无过程奖励；研究时注入可信、原始 token 对齐的步骤奖励 |

状态级优势与步骤级环境奖励是不同变量。关闭状态项并不移除过程奖励；切换 sparse/dense 也不是步骤奖励消融。建议分别研究联合优势权重和过程奖励开关，避免同时改变两个因素。

过程奖励钩子接收可信任务行和完整 episode，返回从零开始的回合索引到奖励向量的映射。向量须与该回合原始生成 token 等长；若研究每步骤一个标量，可将它放在该回合最后一个生成 token，其余位置设零。标准 runner 校验并导出 process_rewards，GiGPO 将其纳入回报，PPO 的 GAE 也会消费这些奖励。该钩子供后续研究代码调用，当前 CLI 不提供未经定义的奖励模型，也不修改原始购物成功标准。

算法对照应固定模型、LoRA rank、数据/源码版本、goal 指纹、任务与环境 seed、每目标候选数、交互预算、评估 seed 和训练目标量；各算法分别训练并使用自己的 actor checkpoint 测试。记录多 seed 的结果及波动，同时记录采样 token、优化次数和成本。PPO 额外训练 critic；GiGPO 状态权重为零仍保留所选 GiGPO 归一化设置，不自动等价于默认 GRPO。

## 安装与准备

从仓库根目录执行，使用 Python 3.12 和 JDK 11。JAVA_HOME 应指向该 JDK，java -version 应可运行。模拟器在本机运行文本环境，需要 Java 搜索索引和 spaCy 模型，无需启动 HTTP 网站或 Chrome。

```bash
uv sync --extra ppo
uv pip install --python .venv/bin/python -r examples/gigpo/requirements-webshop.txt
.venv/bin/python -m spacy download en_core_web_sm

# 直接检出原始 WebShop 模拟环境并固定版本。
git clone https://github.com/princeton-nlp/WebShop.git .cache/webshop
git -C .cache/webshop checkout 64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd

# 复用默认 HF 缓存并构建严格对应的 1k 索引。
.venv/bin/python -m examples.gigpo.data_prepare
```

缓存缺文件时给准备命令增加 --download，补齐所需的三个文件。源码和索引位于仓库忽略的 .cache；也可在 environment.source_dir 指定同版本的已有源码。准备程序不执行上游 setup.sh，不覆盖不同内容的已有数据或未知索引；成功后记录资产 SHA-256 和源码版本，匹配时可重复执行。

PyTRIO 服务配置沿用[项目 README](../../README.md)。模型名称必须是服务支持的 ID，可通过 --model 覆盖。当前案例依赖已有 GiGPO 框架接口提交 9b35900；若接口尚未合入 main，合并顺序为先接口、再案例。

## 运行

先自检真实环境，再验证一个目标的完整采样组，最后训练：

```bash
.venv/bin/python -m examples.gigpo.train --mode check
.venv/bin/python -m examples.gigpo.train --mode probe

# 默认 GRPO：100 个 rollout 批次，每批 16 个目标，每目标 8 条独立候选。
# 共消费 1,600 个训练目标、采样 12,800 条轨迹；结束时完整验证一次。
JAVA_HOME="$PWD/.cache/webshop-java" PYTHON=.venv/bin/python \
  bash examples/gigpo/scripts/run-grpo-webshop.sh

# 普通 GiGPO：使用原配置从头训练，不加载 checkpoint。
JAVA_HOME="$PWD/.cache/webshop-java" PYTHON=.venv/bin/python \
  bash examples/gigpo/scripts/run-gigpo-webshop.sh

# 续训 GiGPO：从旧 step-50 checkpoint 恢复 optimizer，再更新 50 次到 step 100。
JAVA_HOME="$PWD/.cache/webshop-java" PYTHON=.venv/bin/python \
  bash examples/gigpo/scripts/run-gigpo-webshop-resume.sh

# 仅移除状态级优势，其他 GiGPO 配置保持一致。
PYTHON=.venv/bin/python bash examples/gigpo/scripts/run-gigpo-webshop.sh \
  --step-advantage-weight 0 --output-dir output/gigpo-no-state-seed42
```

续训配置与 `run-20261029_133246-z1rzzas6` 对齐：Qwen3.5-4B、LoRA rank 8、batch 8、每目标 8 个候选、单 minibatch、max_turns 20、关闭 thinking，以及原有 GiGPO 优势和 reference KL。目标 max_steps 为 100，因此从 step 50 再执行 50 次更新。旧 run 实际训练到 step 62，但最近保存的 checkpoint 是 step 50，所以 step 51–62 会重新训练。

旧 run 没有本地完整 manifest，所以这次属于兼容续训：恢复远端 optimizer state、权重、步数、数据位置和可确定的 Agent seed offset；无法恢复旧进程的 RNG、SwanLab 聚合窗口和先前 update_history。新输出目录不会覆盖旧日志，并从 step 75 开始保存完整本地 manifest；之后可用标准 checkpoint.resume_from 精确续训。

例如 step 75 后再次中断时，删除配置中的 continuation 段，并设置：

```yaml
trainer:
  checkpoint:
    save_every: 25
    state_dir: ../../../output/webshop-gigpo-qwen3.5-4b-resume-step50-to100-seed42/states
    resume_from: ../../../output/webshop-gigpo-qwen3.5-4b-resume-step50-to100-seed42/states/ppo-step-00000075.json
```

此时框架会从 manifest 恢复远端 optimizer、本地数据位置、RNG、Agent seed、指标计数和更新历史；max_steps 仍表示总目标步数。

check 不连接训练服务；probe 采样并导出轨迹，不执行参数更新。正式配置只在训练结束时完整评估；每 25 步保存 checkpoint。SwanLab 使用单独的 continuation run 和 tags。

独立测试基础模型和训练后的 LoRA 策略：

```bash
# 不传 checkpoint：测试训练前基础策略。
.venv/bin/python -m examples.gigpo.evaluation --algorithm gigpo --output-dir output/eval-base

# 先把训练报告 checkpoint 字段的真实云端 URI 赋给 ACTOR_LORA_URI。
# 该 URI 是 actor 的 save_state 结果，既不是基础模型名，也不是本地文件目录。
.venv/bin/python -m examples.gigpo.evaluation --algorithm gigpo --checkpoint "$ACTOR_LORA_URI" --output-dir output/eval-gigpo
.venv/bin/python -m examples.gigpo.evaluation --algorithm grpo --checkpoint "$GRPO_ACTOR_LORA_URI" --output-dir output/eval-grpo
.venv/bin/python -m examples.gigpo.evaluation --algorithm ppo --checkpoint "$PPO_ACTOR_LORA_URI" --output-dir output/eval-ppo

# 默认配置已使用 official 协议；使用与训练相同的配置运行独立最终测试。
.venv/bin/python -m examples.gigpo.evaluation \
  --config examples/gigpo/configs/gigpo-webshop-configs.yaml \
  --eval-split test --checkpoint "$ACTOR_LORA_URI" --output-dir output/final-test

# 统一训练入口也可执行同样的 checkpoint 评估。
PYTHON=.venv/bin/python bash examples/gigpo/scripts/run-gigpo-webshop.sh \
  --mode evaluate --checkpoint "$ACTOR_LORA_URI" --output-dir output/eval-checkpoint
```

训练后测试先根据云端 URI 加载 actor LoRA 状态，再导出固定采样快照，所有测试批使用该策略。加载错误会终止，不退回基础模型；PPO 测试不需要 critic 权重。报告标明 base_model 或 cloud_lora、实际传入的 actor URI，以及 actor/critic 的零优化更新数。独立评估的 `--checkpoint` 仍只加载策略；训练兼容续训由配置中的 continuation 字段控制。

配置中的相对路径相对于 YAML 目录；--output-dir 相对于当前工作目录。每组实验应使用独立输出目录，保存 episodes.jsonl、框架指标及模式/划分报告，避免覆盖。报告包含算法、全部有效配置、目标范围与实际评估索引、goal SHA-256、数据指纹、源码版本、seed 和 checkpoint。基础模型 ID 应与产生该 LoRA 的实验保持一致。

默认 data.eval_limit=null 覆盖完整目标区间。调试时可设正整数，它也会限制 test，报告会保留实际子集大小及索引。关注成功率、原始 task_score、pass@1、系统失败率和 token 用量；基础设施失败保留在成功率分母中，不能把子集指标或失败采样当作完整基准结果。

独立评估通过 evaluation.py 直接汇总全部内存 episode，在返回报告的 metrics 和框架指标日志中记录 eval/agent/task_score_mean（原始平均分，0–1）及 eval/agent/score（前者乘 100，对应论文 Table 1 的 Score）。例如三个任务原始得分为 1、0.5、0，Score 为 50；它不同于只判断完全成功的 success_rate，也不等于稀疏训练奖励 reward_mean。系统失败且没有评分的尝试按零计分并保留在分母，eval/agent/task_score_count 和 eval/agent/unscored_episodes 分别报告有评分与无评分的尝试数。已评分记录缺失 task_score 或分值异常时评估报错。该汇总不依赖磁盘 episodes.jsonl，历史实验不会自动补算。

离线验证案例接入、锚点锁定、失败清理、消融和 LoRA 加载路径：

```bash
.venv/bin/python -m unittest discover -s tests -p test_webshop_gigpo_example.py

```

离线测试替换 WebShop 和云端服务边界，使用真实 runner、轨迹导出器和训练损失；它不替代真实环境自检、云端训练及论文性能评测。

本次本地接入验证使用上述固定版本、environment.seed=42 和 HF 缓存中的 1k 商品：索引构建成功，生成 6,910 个目标，真实环境自检通过。使用目标对应的商品与选项执行确定性购买得到原始 task_score=1.0；同目标双会话互不影响，终局页面保留，重复初始化的完整 goal 指纹一致。该检查用于确认模拟器接入，不是模型训练后的性能结果。
