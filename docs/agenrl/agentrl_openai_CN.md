# OpenAI 接口的 Agent 与 PyTRIO 训练

[English](agentrl_openai.md)

本文介绍如何通过 Chat Completions 或 Responses 运行使用工具的 Agent，以及如何将已有 SDK 代码接入 PyTRIO 训练。基础训练步骤见 [README](README_CN.md)。

## 选择执行路径

| 目标 | 组件 | 结果 |
| --- | --- | --- |
| 通过外部服务运行工具 | Agent + OpenAIClient | 最终答案和 session trace |
| 使用当前 PyTRIO 策略训练 | PyTRIOBackend + OpenAIEpisodeExecutor + AgentTrainingLoop | 原生 episode 和 PPO/GRPO 轨迹 |
| 复用已有的异步 SDK Agent | 带 execute 的 OpenAIWorkflow | 回调输出和 session trace；训练时再接入 PyTRIO 桥接层 |
| 转发本地 HTTP 请求 | OpenAIProxyServer | API 响应和 session trace |

Agent.run 和 OpenAIWorkflow.run 导出的是用于诊断的 API trace，其 trainable=False，通过 PyTRIO 采样时也一样。训练轨迹需要由 OpenAIEpisodeExecutor 桥接层构造，不能通过编码响应文本来补造输出 token ID 或行为策略 log probability。

## 注册并运行工具

### 安装依赖

在仓库根目录执行：

```bash
uv sync --locked --extra openai
```

训练时还需要安装 ppo。第三方 SDK 回调和本地 HTTP 转发需要 openai-proxy，其中包含 FastAPI：

```bash
uv sync --locked --extra ppo --extra openai-proxy
```

### 定义 Python 工具

装饰器从类型注解生成参数 schema，并从 docstring 的第一段提取描述。后续示例复用以下 add 工具：

```python
from triotune.agentrl import function_tool


@function_tool(retry_safe=True)
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b
```

| 工具声明 | 行为 |
| --- | --- |
| 同步或异步函数 | 均支持；同步函数在 worker 线程执行 |
| JSON 标量、list[T]、dict[str, T]、union、Literal、可选默认值 | 转换为工具 schema |
| 可变参数、仅限位置参数或不支持的注解 | 注册时拒绝 |
| 无效的模型参数 | 返回工具错误观察，不会静默转换类型 |
| 工具实现故障 | 抛出 ToolSystemError，不作为模型答错处理 |

retry_safe 声明工具可以安全重复执行，本身不会启用重试，详见 [限制与重试](#配置限制与重试)。

### 运行 Chat Completions 或 Responses

使用外部服务时，配置 OPENAI_API_KEY，并把 OPENAI_MODEL 设置为可访问的模型。SDK 从环境变量读取 key。

```python
import asyncio
import os

from triotune.agentrl.integrations.openai import Agent


async def run_tools(model, *, api="chat_completions", sdk=None):
    async with Agent(
        name="calculator",
        instructions="Use add to solve the problem. Answer with the number only.",
        tools=[add],
        model=model,
        client=sdk,
        api=api,
        max_turns=5,
        tool_timeout=10,
    ) as agent:
        result = await agent.run("What is 2 + 3?")
        print(result.text, result.session_id)
        return result


if __name__ == "__main__":
    asyncio.run(run_tools(os.environ["OPENAI_MODEL"]))
```

调用 run_tools(model, api="responses") 即可使用 Responses。Agent 执行本地函数，并通过原始 tool-call ID 返回工具结果。使用 Responses 继续对话时，会保留全部 output item，包括 reasoning。

每次运行需要新的 session。可以复用 Agent 执行独立对话，但并发运行不能共享同一个 session。

### 注入自己的服务客户端

在注入的 SDK client 上配置身份验证、请求超时和 API 重试：

```python
from openai import AsyncOpenAI


async def run_with_client(model):
    async with AsyncOpenAI(timeout=60.0, max_retries=2) as sdk:
        return await run_tools(model, sdk=sdk)
```

OpenAIClient 接受同步 OpenAI 或 AsyncOpenAI client。注入的 client 由调用方管理；Agent 内部创建的 client 由其异步上下文管理器关闭。

| 资源 | 负责关闭的一方 |
| --- | --- |
| 传入 Agent 或 OpenAIClient 的 SDK client | 创建该 client 的应用 |
| Agent 内部创建的 SDK client | Agent.close 或其异步上下文管理器 |
| 训练工厂返回的新 workflow、Agent 和 client | OpenAIEpisodeExecutor |
| 训练快照中的 sampling client 和 tokenizer | trainer；关闭 adapter 不会关闭它们 |
| 绑定 session 的本地 SDK transport | OpenAIWorkflow；直接使用时由调用方关闭 |

## 通过 PyTRIO 训练 SDK 执行结果

### 将每个候选绑定到采样快照

训练使用 PyTRIO 凭据，不需要 OpenAI API key。为每个候选创建新的 workflow、Agent、client 和 backend：

```python
from triotune.agentrl.integrations.openai import (
    Agent, OpenAIClient, OpenAIWorkflow, PyTRIOBackend,
)

MODEL = "Qwen/Qwen3.5-4B"


def make_workflow(snapshot):
    backend = PyTRIOBackend(snapshot, model=MODEL, max_tokens=1024)
    client = OpenAIClient(backend=backend)
    return OpenAIWorkflow(Agent(
        name="calculator",
        instructions="Use add. Answer with the number only.",
        tools=[add],
        model=MODEL,
        client=client,
    ))
```

snapshot 由 collector 提供。不要替换它，也不要在对话过程中切换共享 client 的 backend 或修改其采样资源。

### 添加评分器与训练循环

scorer 接收完整的原始 row。只有 messages 会发送给 Agent，answer 保留用于评分。

```python
from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    AgentRolloutWorkflow, AgentTrainingLoop, EpisodeLimits, EpisodeScore,
    EpisodeTrajectoryBuilder, OpenAIEpisodeExecutor,
)


class Outcome:
    async def score(self, row, episode):
        success = (
            episode.status == "completed"
            and episode.final_text is not None
            and episode.final_text.strip() == row["answer"]
        )
        return EpisodeScore(float(success), success)


async def train():
    runner = OpenAIEpisodeExecutor(
        make_workflow,
        scorer=Outcome(),
        limits=EpisodeLimits(
            max_turns=5,
            max_tokens_per_turn=1024,
            max_generated_tokens=2048,
        ),
        trajectory_builder=EpisodeTrajectoryBuilder(mode="individual"),
    )
    collector = AgentRolloutWorkflow(runner, max_concurrency=2)
    trainer = PPOTrainer(
        MODEL,
        args=PPOConfig.for_algorithm(
            "grpo", batch_size=1, num_generations=2, max_steps=1,
        ),
        train_dataset=[{"messages": "What is 2 + 3?", "answer": "5"}],
    )
    return await AgentTrainingLoop(trainer, collector).train()
```

在异步入口调用 await train()。需要使用 Responses 时，在 make_workflow 返回的 Agent 上设置 api="responses"。

| 训练行为 | 预期结果 |
| --- | --- |
| 原始 prompt/output ID 和行为 logprob | 原样保留，不重新编码显示文本 |
| 快照、策略、交互和工具身份 | 传递到原生 episode 和轨迹记录 |
| 外部服务、缺少 logprob 或快照变化 | 在优化器执行前拒绝 |
| 模型动作格式错误或在限制内被截断的生成 | 保留原始动作，交给原生部分 episode 评分流程 |
| 基础设施故障或被拒绝的候选 | 不参与训练 |
| 候选 workflow 资源 | 由桥接层关闭；trainer 的 sampling client 和 tokenizer 仍由 trainer 管理 |

终局奖励和成功标记由原生 scorer 负责。SDK reward 提交属于额外贡献，不要在两条路径重复提交同一个终局奖励。SDK 奖励的 source 名称位于 openai/assignment/ 下。导出模式、过程奖励、分组接受、归一化、GAE 和损失加权遵循 [训练数据说明](agentrl_training_CN.md)。

## PyTRIO 上的模型原生工具

PyTRIOBackend 根据快照中的 tokenizer 选择工具语法，不依赖模型名称白名单。新上架模型或本地别名只要使用已支持的语法，通常不需要传入 model_format：

```python
from triotune.agentrl.integrations.openai import (
    Agent, OpenAIClient, OpenAIWorkflow, PyTRIOBackend,
)


def make_native_workflow(snapshot, model):
    backend = PyTRIOBackend(
        snapshot,
        model=model,
        max_tokens=1024,
        template_kwargs={"enable_thinking": False},
    )
    return OpenAIWorkflow(Agent(
        name="calculator",
        instructions="Use add. Answer with the number only.",
        tools=[add],
        model=model,
        client=OpenAIClient(backend=backend),
    ))
```

将该工厂用于 OpenAIEpisodeExecutor。模型标签必须与 backend 绑定的标签一致；adapter 不加载权重，也不创建 sampling client。

内置 Qwen XML 适配器按以下路径处理：

```text
原始 PyTRIO output ID
  → Qwen35ToolParser 读取 reasoning、函数调用和最终文本
  → 转换为 OpenAI 风格的 Chat Completions 或 Responses 对象
  → Agent 执行工具并追加观察
  → 下一次请求扩展原始采样 token 前缀
```

| 原生路径支持 | 原生路径拒绝 |
| --- | --- |
| 文本 Chat Completions 和携带完整历史的 Responses 输入 | 多模态输入，以及由服务端管理的 previous_response_id 续接 |
| tool_choice="auto" 或 "none" 的函数工具 | 强制或 required 工具，以及受约束的 strict 生成 |
| 匹配的 Qwen3.5 模板和经过校验的 schema | 不支持的 schema 关键字和任意 token ID 覆盖 |
| temperature=1、top_p=1 | 改变 PPO 行为策略概率的采样设置 |

无法识别的 tokenizer 模板只保留文本采样能力，会拒绝工具请求。执行工具前会校验参数，包括重复字段、未知函数和非有限数值。解析失败时，session trace 仍保留原始 token ID、logprob 和错误信息。

保持此前的消息、工具 schema 和实际 assistant 输出不变。工具观察必须匹配原始 call ID。续接渲染器只编码新增的观察和模板后缀，不会用重新编码的显示文本替换此前采样动作。


## Session 身份与执行记录

### 绑定 collector 创建的上下文

常规训练由 OpenAIEpisodeExecutor 创建 session。只有实现自定义执行集成时，才需要显式传入 session：

```python
from triotune.agentrl.integrations.openai import OpenAISession


async def record_candidate(workflow, messages, context, snapshot):
    session = OpenAISession(context=context, training=True)
    result = await workflow.run(
        messages,
        context=context,
        session=session,
        snapshot=snapshot,
    )
    return result.export_trajectory()
```

context 必须是 collector 提供的根 ExecutionContext，包含已知 policy_version，并与 snapshot 一致，不能使用 interaction 上下文。该函数返回 API trace；训练数据仍需通过 OpenAIEpisodeExecutor 生成。

传入的 workflow 由调用方管理，使用完后需要关闭其 Agent/client 和 workflow hook；record_candidate 不接管资源。

| 身份 | 读取位置 |
| --- | --- |
| 对话或 session ID | session.session_id |
| 候选和奖励组 | session.rollout_id、session.group_id |
| episode、任务、候选位置和组大小 | session.context |
| 采样策略 | session.context.snapshot_id 和 policy_version |
| 请求和父请求 | interaction.context |

推理 session 自行创建 ID，其 external snapshot 标签不能证明使用过 PyTRIO 采样。不要把 session ID 当作 episode 或 snapshot ID，也不要把身份字段作为模型生成的工具参数传递。

首次 PyTRIO 请求会固定实际的 snapshot、sampler、tokenizer、client、model 和 stop 设置。仅有相同的身份字符串，不能用来替换这些资源，也不能在同一个 session 中混用外部服务记录和 PyTRIO 采样记录。

### 查看请求与响应

```python
def inspect_requests(session):
    for item in session.interactions:
        print(item.request_id, item.response_id, item.status, item.latency)
        print(item.input_ids, item.output_ids, item.logprobs)
        resolved = session.get_interaction(request_id=item.request_id)
        assert resolved.interaction_id == item.interaction_id
```

外部服务不一定提供原始 token ID 或完整 logprob，缺失字段会保留为空。服务端 HTTP request ID 与 completion/response ID 是不同字段；get_interaction(response_id=...) 用于查找后者。

父请求推断要求历史前缀保持不变，并包含之前的请求及其实际输出。历史存在歧义时，需要在 client 调用中提供 parent_interaction_id，或在 HTTP 请求中提供 X-Triotune-Parent-Interaction-Id。未知、未完成、冲突或跨 session 的父请求会被拒绝。

外部 Responses 可通过 previous_response_id 指定续接；只包含增量输入的请求，不会在后续自动推断时被当作完整前缀。

## Session 生命周期与奖励

### 结束、导出与释放

Agent.run 和 OpenAIWorkflow.run 会自动结束 session。直接通过 OpenAIClient 或 proxy 成功发出的请求，会让 session 保持 open 以便继续；应用在对话结束后负责结束 session：

```python
from triotune.agentrl.integrations.openai import OpenAISession


async def direct_request(adapter, model):
    session = OpenAISession()
    response = await adapter.create_chat_completion(
        model=model,
        messages=[{"role": "user", "content": "Reply with 5."}],
        session=session,
        request_id="answer-request",
    )
    session.finish("completed")
    return session, response
```

| Session 状态 | 允许的操作 |
| --- | --- |
| open | 启动请求，同一个 session 内不允许并发请求 |
| completed 且空闲 | 提交奖励、导出或释放 |
| failed/cancelled 且空闲 | 导出或释放，不能提交奖励 |
| released | 读取或导出 trace，不能继续写入 |

导出和释放前，不能有尚未结束的请求或 Agent 工作。重复设置相同的结束状态没有副作用；不能重新打开 session 或更改结束结果。已注册请求失败或被取消时，会保留部分 trace。

### 提交终局或过程奖励

使用已完成且尚未释放的 session，例如 run_tools 返回的 result.session：

```python
from triotune.agentrl import EpisodeScore, RewardAssignment


def score_session(session, success):
    outcome = EpisodeScore(float(success), success)
    session.submit_reward(outcome, submission_id="outcome-v1")
    session.submit_reward(outcome, submission_id="outcome-v1")  # 完全相同的重试。

    first = session.interactions[0]
    session.submit_reward(
        RewardAssignment("interaction", 0.1, source="process"),
        interaction_id=first.interaction_id,
        submission_id="process-v1",
    )
    trace = session.export_trajectory()
    session.release()
    return trace
```

| 奖励形式 | 定位规则 |
| --- | --- |
| EpisodeScore | session 级终局结果，不能提供显式目标 ID |
| trajectory assignment | session 范围 |
| interaction/turn assignment | 从 0 开始的请求位置，或 interaction_id/request_id/response_id |
| token assignment | (请求索引, 原始输出 token 偏移) |
| 旧式标量 | 默认作用于最后一次交互，也可显式指定目标 |

存在精确 PyTRIO output ID 时，token 偏移和过程奖励向量必须与原始输出匹配，包括 reasoning、工具语法和结束标记。同时提供 position 和 ID 时，两者必须指向同一请求。嵌套目标无效会导致整个提交被拒绝，奖励和 submission ID 都不会被写入。

只有 payload 和目标完全相同时，重复 submission ID 才具有幂等性。内容或目标变化会抛出契约错误。reward、aggregation、discount 和 advantage mode 会被保留；session trace 不执行 PPO 聚合或 GAE。interaction 奖励字段和 event binding 是提交的描述，不能再次累加为额外奖励。

### 配置限制与重试

分别设置模型请求、工具和完整 episode 的边界：

```python
from triotune.agentrl import RetryPolicy
from triotune.agentrl.integrations.openai import Agent


def make_bounded_agent(client, model):
    return Agent(
        name="calculator",
        instructions="Use add. Answer with the number only.",
        tools=[add],
        model=model,
        client=client,
        max_turns=5,
        tool_timeout=10,
        episode_timeout=120,
        max_generated_tokens=2048,
        tool_retry=RetryPolicy(max_attempts=2, retry_tools=True),
    )
```

| 边界 | 负责的一方 |
| --- | --- |
| 外部服务请求超时和 API 重试 | 注入的服务 SDK client |
| 工具超时和显式启用的安全重试 | Agent 的 tool_timeout 和 tool_retry |
| 总截止时间和生成 token 预算 | session，由 Agent 或 OpenAIWorkflow.run 配置 |
| 训练的 token、上下文、请求次数和时间限制 | OpenAIEpisodeExecutor 的 EpisodeLimits |

存在原始 output ID 时，生成 token 预算按原始输出计数，否则使用服务返回的 usage。总截止时间同时覆盖请求和工具循环。

工具重试需要同时满足 retry_tools=True、工具声明 retry_safe=True，并且故障属于可重试的临时错误。默认不重试工具。API 重试不会重放已完成工具或重新运行对话。取消同步工具的 await，不能终止已经启动的 worker 线程。

SDK 评分 hook 可通过 OpenAIWorkflow 的 scorer、interaction_scorer、scorer_timeout、scorer_retry 和 accept_sample 配置。SDK scorer 接收 (input, result)，interaction scorer 接收 (input, result, interaction)。接受判断接收 (result, None)，返回 False 会拒绝结果，不会创建 PPO 行。复用原生 EpisodeScorer 应通过 OpenAIEpisodeExecutor。

## 回放与恢复

### 检查、重新评分或比较已记录的 session

下列操作不会采样模型，也不会创建 PPO 训练数据：

```python
from triotune.agentrl.integrations.openai import OpenAISession


async def inspect_recording(session, other_session):
    observations = await session.replay_tools()
    budget_score = await session.rescore(
        lambda row, trace: float(len(trace.interactions) <= row["max_requests"]),
        {"max_requests": 4},
    )
    suffix = session.branch(session.interactions[0].interaction_id)
    comparison = OpenAISession.compare(session, other_session)
    return observations, budget_score, suffix, comparison
```

| 操作 | 结果 |
| --- | --- |
| replay_tools() | 返回脱离原记录的工具调用和结果，不执行工具 |
| replay_tools(executor) | 使用指定 executor 重新执行已记录的调用 |
| rescore(scorer, row) | 返回新分数，不追加奖励 |
| branch(interaction_id) | 返回已记录的后缀副本，不是 EpisodeBranch，也不是新采样的路径 |
| compare(session_a, session_b) | 比较可用的原始 output ID 和 logprob |

需要重新执行安全工具时，可复用工具注册表：

```python
from triotune.agentrl import FunctionToolExecutor


async def replay_calculator(session):
    return await session.replay_tools(FunctionToolExecutor([add]))
```

传入 executor 会真正再次执行函数。如果重复副作用不安全，使用不带 executor 的 replay_tools() 查看记录。两种方式都不修改原 session。

### 取消后的恢复

先取消并等待拥有该请求的任务结束，再执行恢复。已完成的工具结果保留在 trace 中，仍在运行的交互会被标记为 cancelled：

```python
import asyncio


async def cancel_and_export(task, session):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    if session.status == "completed":
        return session.export_trajectory()
    return session.recover_inflight()
```

在释放 session 前使用该函数。恢复不会重新发起请求或重放工具。failed/cancelled session 不接受延迟奖励；需要再次尝试时，启动新的候选。

## 第三方 Agent 的标准 SDK 入口

### 保留已有 SDK 工具循环

OpenAIWorkflow 可按 execute(client, input) 调用异步函数。client 是真正绑定 session 的 AsyncOpenAI，通过进程内 ASGI transport 通信，不需要监听服务器，也不依赖 Triotune Agent 循环。先安装 openai-proxy。

下面的回调复用同一个 add 工具注册表，并自行控制请求和工具循环：

```python
import copy
import json

from triotune.agentrl import FunctionToolExecutor, ToolCall
from triotune.agentrl.integrations.openai import OpenAIWorkflow


async def execute(client, input):
    messages = (
        [{"role": "user", "content": input}]
        if isinstance(input, str) else copy.deepcopy(input)
    )
    executor = FunctionToolExecutor([add])
    for _ in range(5):
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=executor.schemas,
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return message.content
        for call in message.tool_calls:
            result = await executor.execute(ToolCall(
                call.id, call.function.name, json.loads(call.function.arguments),
            ))
            messages.append({
                "role": "tool", "tool_call_id": call.id, "content": result.content,
            })
    raise RuntimeError("Agent exceeded its request limit")


async def run_existing_agent(adapter, question):
    async with OpenAIWorkflow(client=adapter) as workflow:
        result = await workflow.run(
            question,
            execute=execute,
            episode_timeout=120,
            max_generated_tokens=2048,
        )
        return result.output, result.export_trajectory()
```

MODEL 定义在前面的 PyTRIO 示例中；使用外部 adapter 时，应替换为对应服务的模型标签。adapter 及其注入的上游 client 由调用方管理。workflow 负责结束 session 并关闭本地 SDK transport，回调输出保存在 result.output 中。

回调在后续 Chat tool 消息或 Responses function_call_output 中返回工具结果时，才会记录工具观察。需要保留 call ID 和 assistant 输出历史。没有提交的副作用和工具延迟无法从 SDK 消息推断；回调返回前应等待其请求完成。

### 通过 PyTRIO 训练已有回调

使用只包含 client 的 workflow 工厂、相同的原生 scorer，并把 execute 传给训练桥接层：

```python
from triotune.agentrl import AgentRolloutWorkflow, AgentTrainingLoop, OpenAIEpisodeExecutor
from triotune.agentrl.integrations.openai import OpenAIClient, OpenAIWorkflow, PyTRIOBackend


def make_callback_workflow(snapshot):
    return OpenAIWorkflow(client=OpenAIClient(
        backend=PyTRIOBackend(snapshot, model=MODEL, max_tokens=1024),
    ))


async def train_existing_agent(trainer):
    runner = OpenAIEpisodeExecutor(
        make_callback_workflow,
        execute=execute,
        scorer=Outcome(),
    )
    collector = AgentRolloutWorkflow(runner, max_concurrency=2)
    return await AgentTrainingLoop(trainer, collector).train()
```

trainer 必须使用 MODEL，并提供任务行。原生 Qwen 模板、schema、快照和采样限制仍然适用。绑定 session 的 SDK 请求始终使用 max_retries=0，外部服务的重试策略应配置在上游 SDK client 上。不要覆盖本地 SDK 工厂管理的 base_url 或 http_client。可通过 OpenAIWorkflow.run 的 client_options 传入可选 SDK 设置。

手动执行 SDK 代码时，使用 adapter.as_async_openai(session=session) 或 proxy.create_client(session=session)。关闭该 client 不会结束或释放 session；应在提交奖励或导出前调用 session.finish，完成后再 release。通过 with_options 创建的副本仍绑定同一个 session。

直接 OpenAIClient 请求支持 request_id。session 保持 open 时，重复提交完全相同的已完成 request ID 会返回已有响应，不再调用上游服务。payload、endpoint 或父请求变化，以及仍在运行或已经失败的请求，都不能复用该身份。

## 可选的本地 HTTP 转发

### 创建应用并发起本地请求

proxy 转发 POST /v1/chat/completions 和 POST /v1/responses。工具由调用方执行，并由调用方提交后续请求；proxy 本身不执行工具。

```python
import httpx

from triotune.agentrl.integrations.openai import OpenAIClient, OpenAIProxyServer


def make_proxy(upstream_sdk, model):
    return OpenAIProxyServer(
        OpenAIClient(upstream_sdk, model=model),
        max_sessions=128,
        max_released_session_ids=128,
    )


async def call_local_proxy(proxy, model):
    app = proxy.create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://triotune.local",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Triotune-Session-Id": "demo-session"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with 5."}],
            },
        )
        response.raise_for_status()
    session_id = response.headers["X-Triotune-Session-Id"]
    proxy.finish_session(session_id)
    trace = proxy.sessions[session_id].export_trajectory()
    proxy.release_session(session_id)
    return response.json(), trace
```

该示例使用进程内连接。upstream_sdk 由调用方管理，并在 proxy 工作完成后关闭。

| Header | 用途 |
| --- | --- |
| X-Triotune-Session-Id | 传入对话 ID，或从响应读取分配的 ID |
| X-Triotune-Request-Id | 可选的调用方请求身份 |
| X-Triotune-Parent-Interaction-Id | 可选的显式续接父交互 |
| X-Client-Request-Id | 响应返回的本地 request ID |
| X-Triotune-Interaction-Id | 响应返回的已记录 interaction ID |

### 在 proxy 边界绑定训练 session

PyTRIOBackend 会拒绝默认推理 session。应提供使用 collector 上下文的工厂，而不是根据 HTTP 输入构造训练身份：

```python
from triotune.agentrl import AgentContractError
from triotune.agentrl.integrations.openai import (
    OpenAIClient, OpenAIProxyServer, OpenAISession, PyTRIOBackend,
)


def make_training_proxy(snapshot, contexts):
    def session_factory(session_id):
        if session_id not in contexts:
            raise AgentContractError("Unknown collector session")
        return OpenAISession(
            session_id=session_id, context=contexts[session_id], training=True,
        )

    return OpenAIProxyServer(
        OpenAIClient(backend=PyTRIOBackend(snapshot, model=MODEL)),
        session_factory=session_factory,
    )
```

contexts 把应用 session ID 映射到已有 collector 根上下文，每个上下文必须与传入的快照匹配。该路径记录固定策略的 API 执行信息；需要通过 OpenAIEpisodeExecutor 将执行结果转换为 PPO 训练数据。

### 处理故障和 session 容量

| 情况 | 行为 |
| --- | --- |
| 已注册请求失败 | 保留失败交互，在响应 header 中返回 session 身份 |
| 上游服务返回 HTTP 错误 | 保留原状态码 |
| 连接失败 / 超时 / 非预期 SDK 故障 | HTTP 502 / 504 / 500 |
| 注册前的无效请求 | 回收新分配的空 session，保留已有执行记录 |
| session 容量已满 | 先结束并释放已完成 session，再分配新的 session |
| 已释放 ID 仍保留在缓存中 | 拒绝后续复用 |

max_sessions 和 max_released_session_ids 默认都是 1024。已释放 ID 使用有界 FIFO 缓存；被淘汰后，同一个字符串可以对应新的 session。需要永久唯一性时，应在应用中实现。保留的 released trace 对象仍然只读。

extra_headers 的 key 必须是非空字符串，value 必须是字符串。本 adapter 不支持 streaming、多个 chat choice、后台 Responses、多模态 Agent、子进程执行或面向公网的服务器部署。
