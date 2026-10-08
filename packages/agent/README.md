# pi-agent-core (Python)

`@earendil-works/pi-agent-core` 的 Python 移植（上游：`packages/agent`）。
本包提供三层能力：

1. **有状态的 `Agent`** —— 持有会话记录、派发生命周期事件、执行工具，并提供
   steering / follow-up 消息队列。绝大多数应用只需要这一层。
2. **底层 agent 循环** —— `agent_loop` / `run_agent_loop` 等无状态函数，
   适合自行管理会话状态的宿主程序。
3. **代理 stream 函数** —— `stream_proxy`，让 LLM 请求经由服务器中转而非直连
   provider。

依赖：`pi-ai`（消息类型、stream 协议、provider 目录）与 `httpx`（proxy 传输）。
要求 Python >= 3.14。在本仓库根目录执行 `uv sync` 即可安装。

## 快速开始

```python
import asyncio

from pi_ai import create_models
from pi_ai.providers.all import all_providers
from pi_agent import Agent, AgentInitialState


async def main() -> None:
    # 1. 构建模型目录
    models = create_models()
    for provider in all_providers():
        models.set_provider(provider)
    model = models.get_model("anthropic", "claude-fable-5")
    assert model is not None

    # 2. 创建 agent。models.stream_simple 满足 StreamFn 协议，可直接作为 stream_fn
    agent = Agent(
        stream_fn=models.stream_simple,
        initial_state=AgentInitialState(
            system_prompt="你是一个乐于助人的助手。",
            model=model,
        ),
    )

    # 3. 发起对话；await 返回时本轮（含工具执行）已经结束
    await agent.prompt("用一句话介绍你自己。")

    # 4. 从状态里读取结果
    for message in agent.state.messages:
        print(message.role)

asyncio.run(main())
```

API key 走 `pi-ai` 的凭据解析：导出对应环境变量（如 `ANTHROPIC_API_KEY`、
`OPENAI_API_KEY`、`GEMINI_API_KEY`）即可；也可以用
`AgentOptions.get_api_key` 回调在每次请求前动态解析（适用于会过期的 token）。

## 目录

- [创建 Agent](#创建-agent)
- [发起对话与生命周期](#发起对话与生命周期)
- [事件订阅](#事件订阅)
- [定义工具](#定义工具)
- [消息队列：steering 与 follow-up](#消息队列steering-与-follow-up)
- [回调钩子](#回调钩子)
- [配置项速查](#配置项速查)
- [错误处理](#错误处理)
- [底层循环 API](#底层循环-api)
- [代理（经服务器中转）](#代理经服务器中转)
- [模块对照表](#模块对照表)
- [运行测试](#运行测试)

## 创建 Agent

`Agent` 的构造签名是 `Agent(options=None, **overrides)`：
`options` 可以是 `AgentOptions`、普通 `dict` 或省略；`**overrides` 中的关键字会
覆盖 `options` 的同名字段。

```python
from pi_agent import Agent, AgentInitialState, AgentOptions

# 推荐：显式 AgentOptions
agent = Agent(
    AgentOptions(
        initial_state=AgentInitialState(
            system_prompt="You are helpful.",
            model=model,
            thinking_level="low",
            tools=[my_tool],
        ),
        stream_fn=models.stream_simple,
        steering_mode="one-at-a-time",
    )
)

# 等价的关键字写法
agent = Agent(stream_fn=models.stream_simple, initial_state=AgentInitialState(model=model))
```

### `AgentInitialState`

初始状态，对应上游的 `initialState`：

| 字段 | 说明 |
| --- | --- |
| `system_prompt` | 初始 system prompt；会与 `tools` 一起并入首条 system 消息 |
| `model` | 初始模型（`pi_ai.types.Model`）。省略时是一个 `id="unknown"` 的占位符，**实际使用请显式传入** |
| `thinking_level` | 初始 thinking 档位：`"off"`（默认）/`"minimal"`/`"low"`/`"medium"`/`"high"`/`"xhigh"`/`"max"` |
| `tools` | 初始工具列表（`AgentTool`），会作为工具声明并入首条 system 消息 |
| `messages` | 初始会话记录；若首条不是 system 消息，`system_prompt` 与 `tools` 仍会补插为首条 |

之后修改 system prompt 的方式是**追加**带 `content` / `sections` 的 system 消息
（`agent.state.system_prompt` 是从记录回放的只读属性）；替换工具集则直接赋值
`agent.state.tools = [...]`，循环会在下一次请求前生成 `tools_added` /
`tools_removed` 声明。

### stream 函数（`stream_fn`）三种配置方式

1. **每个 Agent 单独指定**（最常用）：

   ```python
   agent = Agent(stream_fn=models.stream_simple)
   ```

2. **注册进程级默认值**，此后所有未指定 `stream_fn` 的 Agent 都使用它：

   ```python
   from pi_agent import set_default_stream_fn

   set_default_stream_fn(models.stream_simple)   # 传 None 清除
   ```

   未指定且未注册默认值时，首次运行会抛出
   `RuntimeError: No default stream function configured...`。

3. **自定义实现**。`StreamFn` 协议的签名为：

   ```python
   def stream_fn(
       model: Model,
       context: TranscriptContext,          # 已规范化的会话记录（含 system 与工具声明）
       options: SimpleStreamOptions | None,  # 本循环携带 signal、api_key 等
   ) -> AssistantMessageEventStream | Awaitable[AssistantMessageEventStream]:
       ...
   ```

   契约：**不得抛出异常**；失败必须编码进返回的 stream —— 以
   `stop_reason` 为 `"error"` / `"aborted"` 且带 `error_message` 的最终
   `AssistantMessage` 结束。测试里的 `ScriptedProvider`
   （`tests/e2e_test.py`）是一个可参考的最小实现。

## 发起对话与生命周期

```python
await agent.prompt("hello")        # str → 自动包装成 UserMessage
await agent.prompt(user_message)   # 单条消息对象
await agent.prompt([msg1, msg2])   # 消息列表
await agent.continue_()            # 从当前会话记录继续（最后一条须是 user/toolResult）
```

| 方法 / 属性 | 说明 |
| --- | --- |
| `await agent.prompt(...)` | 发起新对话；已有运行进行时抛 `RuntimeError`，此时应改用 `steer()` / `follow_up()` |
| `await agent.continue_()` | 续跑：处理排队消息，或让最后一条 `user`/`toolResult` 之后的回合继续；末条是 `assistant` 且无排队消息时抛 `RuntimeError` |
| `await agent.wait_for_idle()` | 等当前运行及所有被 await 的 `agent_end` 监听器结束后返回 |
| `agent.abort()` | 中止当前运行（无运行时是 no-op）；最终 assistant 消息的 `stop_reason` 为 `"aborted"` |
| `agent.reset()` | 清空会话与队列，保留回放得到的 prompt/tool 基线；运行中调用会抛 `RuntimeError` |
| `agent.signal` | 当前运行的 `AbortSignal`（无运行时为 `None`），可用于取消配套任务 |
| `agent.state` | 见下表 |

由于 `prompt()` 会一直 await 到本轮结束，需要中途 `abort()` 时把它包成任务：

```python
task = asyncio.ensure_future(agent.prompt("讲一个很长的故事"))
await asyncio.sleep(0.5)
agent.abort()
await task
```

### `agent.state`（`AgentState`）

| 字段 | 说明 |
| --- | --- |
| `messages` | 会话记录；赋新列表时会复制顶层列表 |
| `tools` | 当前可执行工具，语义同上 |
| `model` / `thinking_level` | 当前模型与 thinking 档位，可直接改 |
| `is_streaming` | 是否有运行在进行（要等 `agent_end` 监听器结算后才变回 `False`） |
| `streaming_message` | 流式期间增长中的部分 assistant 消息；消息结束后为 `None` |
| `pending_tool_calls` | 正在执行的 tool call id 集合 |
| `error_message` | 最近一次失败/中止回合的错误文本（若有） |
| `system_prompt` | 只读；从记录中的 system 消息回放 |

## 事件订阅

```python
def on_event(event, signal):     # 同步或异步均可；signal 是本次运行的 AbortSignal
    print(event.type)

unsubscribe = agent.subscribe(on_event)
unsubscribe()                    # 或 agent.unsubscribe(on_event)
```

监听器按订阅顺序被 await，其耗时计入运行完成判定；`agent_end` 是一次运行的
最后一个事件，但该事件上的监听器结束之前 agent 不会进入空闲。

一次 prompt 的典型事件序列：

```
agent_start → turn_start → message_start → message_update* → message_end
            → (tool_execution_start → tool_execution_update* → tool_execution_end)*
            → (turn_start → ... → turn_end)*   # 每个回合重复
            → turn_end → agent_end
```

| 事件 | 关键字段 | 常见用途 |
| --- | --- | --- |
| `agent_start` / `agent_end` | `agent_end.messages` 为本次运行产出的消息 | 运行边界、空闲判断 |
| `turn_start` / `turn_end` | `turn_end.message`、`turn_end.tool_results` | 回合计数、日志 |
| `message_start` / `message_end` | `message` | 消息入账 |
| `message_update` | `message`（增长中）、`assistant_message_event`（底层 delta） | 流式渲染 |
| `tool_execution_start` | `tool_call_id`、`tool_name`、`args` | 工具开始提示 |
| `tool_execution_update` | `partial_result` | 工具进度条 |
| `tool_execution_end` | `result`、`is_error` | 工具完成提示 |

流式渲染示例：

```python
async def render(event, _signal):
    if event.type == "message_update" and event.message.role == "assistant":
        text = "".join(
            block.text for block in event.message.content if block.type == "text"
        )
        print(f"\r{text}", end="", flush=True)
    elif event.type == "message_end" and event.message.role == "assistant":
        print()

agent.subscribe(render)
await agent.prompt("写一段话")
```

## 定义工具

```python
from pi_ai.types import TextContent
from pi_agent import AgentTool, AgentToolResult

weather_tool = AgentTool(
    name="get_weather",
    label="天气",                      # 供 UI 展示的标签
    description="查询指定城市的天气",   # 给模型看的工具描述
    parameters={                       # JSON Schema（TypeBox 的替代）
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    output_schema=None,                # 可选：structured_content 的 JSON Schema
    execute=execute,                   # 见下
)
```

`execute` 的完整签名是 `execute(tool_call_id, params, signal, on_update)`，
但可以像 TypeScript 一样**只声明前几个参数**，循环按声明个数截取；
同步或异步函数都支持。参数已按 `parameters` schema 校验完毕。

```python
async def execute(tool_call_id: str, args: dict, signal, on_update) -> AgentToolResult:
    # 1. 可选：流式产出部分结果（对应 tool_execution_update 事件）
    on_update(AgentToolResult(content=[TextContent(text="查询中...")]))

    # 2. 可选：响应取消
    # signal 是 AbortSignal，可检查 signal.aborted

    # 3. 返回最终结果
    return AgentToolResult(
        content=[TextContent(text=f"{args['city']}：晴")],  # 返回给模型的文本
        details={"unit": "c"},                             # 供日志/UI 的任意结构化数据
        structured_content={"temp_c": 26},                 # 需要 output_schema 才有意义
        is_error=False,                                    # True 时模型把 content 当错误结果
        terminate=False,                                   # True 时本批工具结束后 agent 停止
    )
```

行为要点：

- **工具故障绝不抛出**：未找到工具、参数校验失败、`execute` 抛异常，都会变成
  `is_error=True` 的错误 tool result，由模型自行解读。
- `terminate=True` 只有在本批**每个**收尾的 tool result 都置为 true 时才提前终止。
- `on_update` 在该 tool call 结算后被忽略，迟到的部分结果不会发出事件。
- 同一条 assistant 消息里的多个 tool call 默认 `parallel` 执行；可用
  `AgentOptions.tool_execution="sequential"` 改为依次执行，或在单个工具上用
  `AgentTool.execution_mode` 覆盖。
- `AgentTool.prepare_arguments` 是 schema 校验前处理原始参数的兼容垫片。
- `AgentTool.replay` 是「持久意图已存在但结果未知」时的恢复策略（对接持久化宿主）。

注册：放进 `AgentInitialState(tools=[...])`，或运行时 `agent.state.tools = [...]`。

## 消息队列：steering 与 follow-up

运行期间入队的消息对象（通常是 `pi_ai.types.UserMessage`），不会立刻进入会话：

```python
from pi_ai.types import TextContent, UserMessage

agent.steer(UserMessage(role="user", content="改用简体中文回答", timestamp=...))
#  在当前 assistant 回合结束后注入，用于「中途纠正方向」

agent.follow_up(UserMessage(role="user", content="再讲一个", timestamp=...))
#  仅在 agent 本将停止时才运行，用于「它停下来之后接着说」
```

| 方法 | 说明 |
| --- | --- |
| `steer(msg)` / `follow_up(msg)` | 入队 |
| `has_queued_messages()` | 两队列是否还有待处理消息 |
| `peek_queued_messages()` | 预览下次 drain 会取出的消息，不消费 |
| `clear_steering_queue()` / `clear_follow_up_queue()` / `clear_all_queues()` | 清空 |

出队模式由 `steering_mode` / `follow_up_mode` 控制（构造时设置，或运行时改
`agent.steering_mode` / `agent.follow_up_mode`）：

- `"one-at-a-time"`（默认）—— 每个 drain 点只取最旧的一条；
- `"all"` —— 每个 drain 点一次取完。

队列里的消息在运行中/运行结束失败时都不会被丢弃，下次
`prompt()` / `continue_()` 会消费。

## 回调钩子

全部可以通过 `AgentOptions` 传入（同步或异步均可）。除特别注明外，钩子故障
会转为错误结果而不是让运行崩溃。

| 钩子 | 时机 | 签名 / 返回值 |
| --- | --- | --- |
| `convert_to_llm(messages)` | 每次 LLM 调用前，把会话记录转成 LLM 消息 | 默认实现只保留 `system`/`user`/`assistant`/`toolResult` 角色；**不得抛出** |
| `transform_context(messages, signal)` | `convert_to_llm` 之前 | 返回替换后的消息列表；**不得抛出** |
| `get_api_key(provider)` | 每次请求前 | 返回 `str \| None`；用于动态 token |
| `on_payload(payload, ...)` / `on_response(...)` | provider 发请求 / 收响应时 | 由 `pi-ai` 适配器调用，可观察或改写 payload |
| `on_provider_stream_event(data, model)` | 收到 provider 原始 stream 事件 | 透传观察 |
| `prepare_request(ctx, signal)` | 每次会话型 provider 请求之前 | `ctx: PrepareRequestContext(context, model, thinking_level)`；返回 `AgentRequestUpdate(context/model/thinking_level)` 覆盖本次请求，或 `None` |
| `before_tool_call(ctx, signal)` | 参数校验完成后、执行前 | `ctx: BeforeToolCallContext`；返回 `BeforeToolCallResult(block=True, reason=..., terminate=...)` 可拦截执行，或 `None` 放行 |
| `after_tool_call(ctx, signal)` | 执行结束后、事件产出前 | `ctx: AfterToolCallContext`；返回 `AfterToolCallResult(content/details/is_error/usage/terminate/structured_content)` 逐字段覆盖结果，或 `None` |
| `finish_turn(turn, signal)` | assistant 回合及全部 tool result 完成后 | 返回 `None`（保持正常调度）、`{"action": "continue"}`（强制再发一次请求）、`{"action": "end"}`（立即停止）或 `AgentTurnDecision` |
| `prepare_next_turn_with_context(turn_ctx, signal)` | `turn_end` 之后、下一回合之前 | 返回 `AgentLoopTurnUpdate(context/messages/model/thinking_level)` 或 `None`（压缩、切换模型等场景） |
| `prepare_next_turn(signal)` | 同上，**旧签名**（只收 signal） | 与上者二选一；同时设置时 `with_context` 版本优先 |

拦截工具示例：

```python
from pi_agent import AgentOptions, BeforeToolCallResult

def require_confirmation(ctx) -> BeforeToolCallResult | None:
    if ctx.tool_call.name in {"delete_file", "send_email"}:
        if not user_approved(ctx.args):
            return BeforeToolCallResult(block=True, reason="用户拒绝了本次操作")
    return None

agent = Agent(before_tool_call=require_confirmation, ...)
```

## 配置项速查

`AgentOptions` 的其余字段：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `session_id` | `None` | 会话标识，转发给支持缓存的 provider 后端 |
| `thinking_budgets` | `None` | 各 thinking 档位对应的 token 预算 |
| `transport` | `"auto"` | 首选传输方式，转发给 stream 函数 |
| `max_retry_delay_ms` | `None` | provider 请求重试延迟上限（毫秒） |
| `tool_execution` | `"parallel"` | 多 tool call 的执行策略：`"parallel"` / `"sequential"` |
| `steering_mode` / `follow_up_mode` | `"one-at-a-time"` | 见「消息队列」 |

## 错误处理

运行期间的任何异常（provider 故障、stream 函数抛错……）都**不会**从
`prompt()` 抛给调用方，而是被转换成一条 `stop_reason="error"`
（中止时为 `"aborted"`）的 assistant 错误消息，并派发完整的事件序列：

```python
await agent.prompt("hello")

last = agent.state.messages[-1]
if last.role == "assistant" and last.stop_reason in ("error", "aborted"):
    print(last.error_message)      # 也同步记录在 agent.state.error_message
```

会主动抛 `RuntimeError` 的只有并发类误用：运行中再次 `prompt()` /
`continue_()` / `reset()`，以及 `continue_()` 遇到空记录或末条为
`assistant` 却没有排队消息。

## 底层循环 API

不使用 `Agent` 时，直接驱动循环，自行管理 `AgentContext` 与事件消费。
注意：`AgentLoopConfig` **必须提供 `convert_to_llm`**（`Agent` 会自动填默认值，
底层调用不会）。

```python
from pi_agent import (
    AgentContext,
    AgentLoopConfig,
    agent_loop,
    agent_loop_continue,
    run_agent_loop,
    run_agent_loop_continue,
    run_tool_call,
    RunToolCallOptions,
)
from pi_agent.agent import default_convert_to_llm

# 方式一：事件流风格 —— 返回可异步迭代的 EventStream，
# 以 agent_end 事件（携带最终消息列表）终止；也可 await stream.result()
stream = agent_loop(
    [user_message],
    AgentContext(messages=[], tools=[weather_tool]),
    AgentLoopConfig(model=model, convert_to_llm=default_convert_to_llm),
)
async for event in stream:
    print(event.type)
final_messages = await stream.result()

# 方式二：await 风格 —— 直接返回本次运行追加的消息，
# emit 可以是同步或异步的 sink
async def sink(event) -> None:
    print(event.type)

new_messages = await run_agent_loop(
    [user_message],
    AgentContext(messages=[], tools=[weather_tool]),
    AgentLoopConfig(model=model, convert_to_llm=default_convert_to_llm),
    sink,
    stream_fn=models.stream_simple,
)

# 续跑变体：不添加新消息，从既有上下文继续
# await run_agent_loop_continue(context, config, sink, ...)
# agent_loop_continue(context, config) -> EventStream
#   最后一条消息（经 convert_to_llm 后）必须是 user 或 toolResult，
#   否则 ValueError
```

单独执行一次 tool call（不产事件、不追加消息，但完整走过
参数准备 → 校验 → `before_tool_call` → 执行 → `after_tool_call`）：

```python
from pi_ai.types import ToolCall

outcome = await run_tool_call(
    ToolCall(id="call-1", name="get_weather", arguments={"city": "上海"}),
    RunToolCallOptions(
        tools=[weather_tool],
        context=AgentContext(messages=..., tools=[weather_tool]),
        assistant_message=assistant_message,   # 传给各 hook 的上下文
        signal=None,
        before_tool_call=None,
        after_tool_call=None,
    ),
)
print(outcome.is_error, outcome.result.content)
```

`run_tool_call` 的故障同样绝不拒绝：未知工具、校验错误、被拦截的调用与
`execute` 抛出的异常都以 `is_error=True` 返回。其他工具在执行中调用本函数，
因此权限检查类 hook 对嵌套调用同样生效。

## 代理（经服务器中转）

`stream_proxy` 用于 LLM 请求不直连 provider、而是发往中转服务器的场景
（服务器会从 delta 事件里剥除 `partial` 字段以省带宽，客户端本地重建）。

由于 agent 循环传给 `stream_fn` 的是普通 `SimpleStreamOptions`，不含
`proxy_url` / `auth_token`，所以实际用法是包一层：

```python
from dataclasses import fields

from pi_agent import ProxySerializableStreamOptions, ProxyStreamOptions, stream_proxy


def make_proxy_stream_fn(*, proxy_url: str, auth_token: str):
    """把 agent 循环的选项转发给 stream_proxy，并附上连接凭据。"""

    def stream_fn(model, context, options=None):
        forwarded = {
            f.name: getattr(options, f.name)
            for f in fields(ProxySerializableStreamOptions)
            if options is not None and hasattr(options, f.name)
        }
        return stream_proxy(
            model,
            context,
            ProxyStreamOptions(
                **forwarded,
                proxy_url=proxy_url,
                auth_token=auth_token,                    # Authorization: Bearer ...
                signal=getattr(options, "signal", None),
            ),
        )

    return stream_fn


agent = Agent(stream_fn=make_proxy_stream_fn(
    proxy_url="https://genai.example.com",
    auth_token="...",
))
```

要点：

- `stream_proxy(model, context, options)` 直接返回
  `ProxyMessageEventStream`，满足 `StreamFn` 协议。
- 发往服务器的是 `build_proxy_request_options` 投影出的可序列化子集
  （`ProxySerializableStreamOptions`）；`signal`、`auth_token`、`proxy_url`、
  `fetch`、`timeout_ms` 仅本地使用。
- `fetch` 可注入自定义传输（测试即用它离线桩掉网络）。
- 代理故障（非 2xx、连接被掐断）一律编码成 `stop_reason="error"` 的事件，
  不会挂起。

也可以脱离 `Agent` 单独使用 `stream_proxy`，以及用 `process_proxy_event`
把单个代理事件还原为 assistant 消息事件（重建 partial 消息）。

## 模块对照表

| Upstream | Here |
| --- | --- |
| `src/types.ts` | `pi_agent/types.py` |
| `src/agent-loop.ts` | `pi_agent/agent_loop.py` |
| `src/agent.ts` | `pi_agent/agent.py` |
| `src/proxy.ts` | `pi_agent/proxy.py` |
| `src/stream-fn.ts` | `pi_agent/stream_fn.py` |

命名与数据模型约定（snake_case 字段 + camelCase JSON alias 等）见仓库根目录的
[PORTING.md](../../PORTING.md)。

## 运行测试

在仓库根目录：

```bash
uv run pytest packages/agent
```

本包测试完全离线（脚本化 stream 函数，不触网）。若本机开了系统代理，本地回放
用例可能报 502，此时先设置 `NO_PROXY`（如 `NO_PROXY=127.0.0.1,localhost`）再跑。
