"""全程使用 :data:`AgentMessage` 的 agent 循环。

移植自 ``packages/agent/src/agent-loop.ts``。循环仅在 LLM 调用边界处把
``AgentMessage`` 转换为 :data:`pi_ai.types.Message`。

回调契约（与 TypeScript 注释一致）：``convert_to_llm``、``transform_context``、
``get_api_key``、``get_steering_messages`` 与 ``get_follow_up_messages``
都不得抛出。抛出的回调会中断循环且不会产出正常事件序列，与 TypeScript 原版相同。
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from pi_ai.types import (
    AssistantMessage,
    Context,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from pi_ai.utils.async_utils import maybe_await
from pi_ai.utils.event_stream import EventStream
from pi_ai.utils.serde import from_json
from pi_ai.utils.transcript import (
    get_current_tools,
    get_tool_state_changes,
    normalize_context,
    to_tool_declaration,
)
from pi_ai.utils.validation import validate_tool_arguments

from .stream_fn import get_default_stream_fn
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentRequestUpdate,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentEventSink,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AgentToolCall,
    AgentToolCallOutcome,
    AgentToolResult,
    AgentTurnContext,
    AgentTurnDecision,
    BeforeToolCallContext,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    PrepareNextTurnContext,
    PrepareRequestContext,
    StreamFn,
    ToolCallHooks,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)

__all__ = [
    "AgentEventSink",
    "RunToolCallOptions",
    "ToolCallHooks",
    "agent_loop",
    "agent_loop_continue",
    "run_agent_loop",
    "run_agent_loop_continue",
    "run_tool_call",
]


def now_ms() -> int:
    """Unix 纪元毫秒数，JavaScript ``Date.now()`` 的移植。"""
    return int(time.time() * 1000)


def create_agent_stream() -> EventStream[AgentEvent, list[AgentMessage]]:
    """:func:`agent_loop` 与 :func:`agent_loop_continue` 返回的事件流。"""
    return EventStream(
        lambda event: event.type == "agent_end",
        lambda event: event.messages if event.type == "agent_end" else [],
    )


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Any = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """用新的 prompt 消息启动 agent 循环。

    prompt 会加入上下文，并为其产出相应事件。

    Args:
        prompts: 要注入上下文的新 prompt 消息列表。
        context: 当前会话上下文，含既有消息与可执行工具集。
        config: agent 循环配置。
        signal: 取消信号；为 ``None`` 时不响应取消。
        stream_fn: 自定义的 stream 函数；为 ``None`` 时使用默认实现。

    Returns:
        以 :class:`AgentEvent` 为事件、以最终消息列表为结束值的 :class:`EventStream`。
    """
    stream = create_agent_stream()

    async def drive() -> None:
        """驱动后端循环，并在结束时用最终消息关闭事件流。"""
        messages = await run_agent_loop(
            prompts,
            context,
            config,
            lambda event: stream.push(event),
            signal,
            stream_fn,
        )
        stream.end(messages)

    task = asyncio.ensure_future(drive())
    task.add_done_callback(_swallow_task_result)
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Any = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """从当前上下文继续 agent 循环，不添加新消息。

    用于重试——上下文中已有 user 消息或 tool 结果。

    **注意：** context 的最后一条消息经 ``convert_to_llm`` 转换后必须是 ``user``
    或 ``toolResult`` 角色，否则 LLM provider 会拒绝请求。此处无法校验，因为
    ``convert_to_llm`` 每回合只调用一次。

    Args:
        context: 要从中继续的会话上下文。
        config: agent 循环配置。
        signal: 取消信号；为 ``None`` 时不响应取消。
        stream_fn: 自定义的 stream 函数；为 ``None`` 时使用默认实现。

    Returns:
        以 :class:`AgentEvent` 为事件、以最终消息列表为结束值的 :class:`EventStream`。

    Raises:
        ValueError: 当 ``context.messages`` 为空，或最后一条消息的角色为 ``assistant`` 时。
    """
    if len(context.messages) == 0:
        raise ValueError("Cannot continue: no messages in context")

    if getattr(context.messages[-1], "role", None) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = create_agent_stream()

    async def drive() -> None:
        """驱动继续循环，并在结束时用最终消息关闭事件流。"""
        messages = await run_agent_loop_continue(
            context,
            config,
            lambda event: stream.push(event),
            signal,
            stream_fn,
        )
        stream.end(messages)

    task = asyncio.ensure_future(drive())
    task.add_done_callback(_swallow_task_result)
    return stream


def _swallow_task_result(task: asyncio.Future[Any]) -> None:
    """报告后台循环的失败，但不结束其事件流。

    TypeScript 原版只挂 ``.then`` 处理器，因此被拒绝的 ``runAgentLoop`` 会让
    事件流悬置，Node 会报告 unhandled rejection。本移植取回异常（避免
    "exception was never retrieved" 警告）并打印它，对应那次报告。

    Args:
        task: 后台驱动循环的 :class:`asyncio.Task`。
    """
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        traceback.print_exception(error)


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """携带新的 prompt 消息运行循环，返回它追加的全部消息。

    Args:
        prompts: 要注入上下文的新 prompt 消息列表。
        context: 当前会话上下文。
        config: agent 循环配置。
        emit: 事件 sink，用于产出 agent 事件。
        signal: 取消信号；为 ``None`` 时不响应取消。
        stream_fn: 自定义的 stream 函数；为 ``None`` 时使用默认实现。

    Returns:
        本次运行追加到上下文的消息列表。
    """
    sink = _wrap_sink(emit)
    initial_messages = declare_tool_changes(context, prompts)
    new_messages: list[AgentMessage] = [*initial_messages]
    current_context = AgentContext(
        messages=[*context.messages, *initial_messages],
        tools=context.tools,
    )

    await sink(AgentStartEvent())
    await sink(TurnStartEvent())
    for message in initial_messages:
        await sink(MessageStartEvent(message=message))
        await sink(MessageEndEvent(message=message))

    await run_loop(
        current_context,
        new_messages,
        config,
        signal,
        sink,
        stream_fn if stream_fn is not None else get_default_stream_fn(),
    )
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """从既有上下文运行循环，只返回新追加的消息。

    Args:
        context: 要从中继续的会话上下文。
        config: agent 循环配置。
        emit: 事件 sink，用于产出 agent 事件。
        signal: 取消信号；为 ``None`` 时不响应取消。
        stream_fn: 自定义的 stream 函数；为 ``None`` 时使用默认实现。

    Returns:
        本次运行追加到上下文的消息列表。

    Raises:
        ValueError: 当 ``context.messages`` 为空，或最后一条消息的角色为 ``assistant`` 时。
    """
    if len(context.messages) == 0:
        raise ValueError("Cannot continue: no messages in context")

    if getattr(context.messages[-1], "role", None) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    sink = _wrap_sink(emit)
    new_messages: list[AgentMessage] = []
    # 与 TypeScript 的 ``{ ...context }`` 展开一样，这里保留调用方的消息数组对象，
    # 因此追加的 assistant/tool 消息在 ``context.messages`` 上也可见。
    current_context = AgentContext(messages=context.messages, tools=context.tools)

    await sink(AgentStartEvent())
    await sink(TurnStartEvent())

    await run_loop(
        current_context,
        new_messages,
        config,
        signal,
        sink,
        stream_fn if stream_fn is not None else get_default_stream_fn(),
    )
    return new_messages


def _wrap_sink(emit: AgentEventSink) -> AgentEventSink:
    """把同步或异步的事件 sink 统一规范化为异步 sink。

    Args:
        emit: 同步或异步的事件 sink。

    Returns:
        统一为异步调用的事件 sink。
    """

    async def sink(event: AgentEvent) -> None:
        """转发单个事件，并按需等待同步或异步 sink。

        Args:
            event: 要转发的事件。
        """
        await maybe_await(emit(event))

    return sink


def _steering(config: AgentLoopConfig) -> Any:
    """取出配置提供的 steering 消息。

    Args:
        config: agent 循环配置。

    Returns:
        steering 消息；未定义 ``get_steering_messages`` 时为空列表。
    """
    if config.get_steering_messages is None:
        return []
    return config.get_steering_messages()


def _follow_ups(config: AgentLoopConfig) -> Any:
    """取出配置提供的 follow-up 消息。

    Args:
        config: agent 循环配置。

    Returns:
        follow-up 消息；未定义 ``get_follow_up_messages`` 时为空列表。
    """
    if config.get_follow_up_messages is None:
        return []
    return config.get_follow_up_messages()


async def run_loop(
    initial_context: AgentContext,
    new_messages: list[AgentMessage],
    initial_config: AgentLoopConfig,
    signal: Any,
    emit: AgentEventSink,
    stream_function: StreamFn,
) -> None:
    """:func:`agent_loop` 与 :func:`agent_loop_continue` 共享的主循环逻辑。

    Args:
        initial_context: 循环起始时的会话上下文。
        new_messages: 用于收集本次运行新增消息的输出列表。
        initial_config: 循环起始时的配置。
        signal: 取消信号；为 ``None`` 时不响应取消。
        emit: 事件 sink。
        stream_function: 实际发起 stream 请求的函数。
    """
    current_context = initial_context
    config = initial_config
    last_completed_turn: PrepareNextTurnContext | None = None
    explicit_continuation = False
    # 启动时先检查 steering 消息（用户可能在等待期间已经输入）
    pending_messages: list[AgentMessage] = list(await maybe_await(_steering(config)) or [])

    # 外层循环：agent 本将停止时，若队列中有 follow-up 消息则继续
    while True:
        has_more_tool_calls = True

        # 内层循环：处理 tool call 与 steering 消息
        while has_more_tool_calls or len(pending_messages) > 0:
            prepared_messages: list[AgentMessage] = []
            if last_completed_turn is not None:
                next_turn_snapshot: AgentLoopTurnUpdate | None = None
                if config.prepare_next_turn is not None:
                    next_turn_snapshot = _coerce_optional(
                        AgentLoopTurnUpdate,
                        await maybe_await(config.prepare_next_turn(last_completed_turn)),
                    )
                if next_turn_snapshot is not None:
                    if next_turn_snapshot.context is not None:
                        current_context = next_turn_snapshot.context
                    prepared_messages = next_turn_snapshot.messages or []
                    config = replace(
                        config,
                        model=next_turn_snapshot.model if next_turn_snapshot.model is not None else config.model,
                        reasoning=_next_reasoning(config.reasoning, next_turn_snapshot.thinking_level),
                    )
                # 准备工作可能耗时较长（例如压缩）。期间排队的 steering 消息要在此捡起。
                # 只有上次轮询返回空时才再次轮询，否则 one-at-a-time 模式会在同一
                # 回合投递两条消息。
                if len(pending_messages) == 0:
                    pending_messages = list(await maybe_await(_steering(config)) or [])
                await emit(TurnStartEvent())

            # 在下一次 assistant 响应之前处理已准备与已排队的消息。
            for message in declare_tool_changes(current_context, [*prepared_messages, *pending_messages]):
                await emit(MessageStartEvent(message=message))
                await emit(MessageEndEvent(message=message))
                current_context.messages.append(message)
                new_messages.append(message)
            pending_messages = []

            request_update = None
            if config.prepare_request is not None:
                request_update = _coerce_optional(
                    AgentRequestUpdate,
                    await maybe_await(
                        config.prepare_request(
                            PrepareRequestContext(
                                context=current_context,
                                model=config.model,
                                thinking_level=config.reasoning if config.reasoning is not None else "off",
                            ),
                            signal,
                        )
                    ),
                )
            if request_update is not None:
                if request_update.context is not None:
                    current_context = request_update.context
                config = replace(
                    config,
                    model=request_update.model if request_update.model is not None else config.model,
                    reasoning=_next_reasoning(config.reasoning, request_update.thinking_level),
                )

            # 流式获取 assistant 响应
            message = await stream_assistant_response(current_context, config, signal, emit, stream_function)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                last_completed_turn = AgentTurnContext(
                    message=message,
                    tool_results=[],
                    context=current_context,
                    new_messages=new_messages,
                )
                if config.finish_turn is not None:
                    await maybe_await(config.finish_turn(last_completed_turn, signal))
                await emit(TurnEndEvent(message=message, tool_results=[]))
                await emit(AgentEndEvent(messages=new_messages))
                return

            # 检查是否有 tool call
            tool_calls = [content for content in message.content if content.type == "toolCall"]

            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                # ``length`` 停止原因表示输出被 token 上限截断，消息中的每个
                # tool call 都可能带着被截断的参数。与其执行可能损坏的调用，
                # 不如全部标记失败。
                if message.stop_reason == "length":
                    executed_tool_batch = await fail_tool_calls_from_truncated_message(tool_calls, emit)
                else:
                    executed_tool_batch = await execute_tool_calls(
                        current_context, message, config, signal, emit
                    )
                tool_results.extend(executed_tool_batch.messages)
                has_more_tool_calls = not executed_tool_batch.terminate

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            last_completed_turn = AgentTurnContext(
                message=message,
                tool_results=tool_results,
                context=current_context,
                new_messages=new_messages,
            )
            decision: AgentTurnDecision | None = None
            if config.finish_turn is not None:
                decision = _normalize_turn_decision(
                    await maybe_await(config.finish_turn(last_completed_turn, signal))
                )
            await emit(TurnEndEvent(message=message, tool_results=tool_results))

            if decision is not None and decision.action == "end":
                await emit(AgentEndEvent(messages=new_messages))
                return

            explicit_continuation = decision is not None and decision.action == "continue"
            pending_messages = list(await maybe_await(_steering(config)) or [])
            if has_more_tool_calls or len(pending_messages) > 0:
                explicit_continuation = False

        # agent 到此本将停止，检查是否有 follow-up 消息。
        follow_up_messages = list(await maybe_await(_follow_ups(config)) or [])
        if len(follow_up_messages) > 0:
            # 置为待处理，让内层循环处理它们
            explicit_continuation = False
            pending_messages = follow_up_messages
            continue

        # 没有自然选中的下一次请求，就用一轮仅上下文的回合兑现 continue 决策。
        if explicit_continuation:
            explicit_continuation = False
            continue

        # 没有更多消息，退出
        break

    await emit(AgentEndEvent(messages=new_messages))


def _coerce_optional(cls: Any, value: Any) -> Any:
    """把 TS 风格的对象字面量（mapping）强制转换为对应 dataclass，真实实例原样返回。

    TypeScript hook 返回如 ``{ action: "end" }``、``{ block: true }`` 的普通对象
    字面量。本移植返回 dataclass，但同样接受字面量形式，使接口与原版一样宽容。
    首个参数为目标 dataclass 类型，按惯例命名为 ``cls``，故不列入下方参数表。

    Args:
        value: 待转换的值，可为 ``None``、目标类型的实例或 mapping。

    Returns:
        转换后的实例；``value`` 为 ``None`` 时返回 ``None``。
    """
    if value is None or isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return from_json(cls, value)
    return value


def _normalize_turn_decision(value: Any) -> AgentTurnDecision | None:
    """接受 :class:`AgentTurnDecision`、TS 风格的 ``{"action": ...}`` mapping 或 ``None``。

    Args:
        value: hook 返回的原始值。

    Returns:
        规范化后的 :class:`AgentTurnDecision`；无法识别时返回 ``None``。
    """
    if value is None:
        return None
    coerced = _coerce_optional(AgentTurnDecision, value)
    action = getattr(coerced, "action", None)
    if action is None:
        return None
    return AgentTurnDecision(action=action)


def _next_reasoning(current: Any, thinking_level: Any) -> Any:
    """``prepare_next_turn`` 与 ``prepare_request`` 共用的 ``thinkingLevel`` 映射规则。

    ``None`` 保持当前 reasoning；``"off"`` 清除；其他值直接替换。

    Args:
        current: 当前 reasoning 设置。
        thinking_level: hook 返回的 thinking level。

    Returns:
        更新后的 reasoning 设置。
    """
    if thinking_level is None:
        return current
    if thinking_level == "off":
        return None
    return thinking_level


def declare_tool_changes(context: AgentContext, pending_messages: list[AgentMessage]) -> list[AgentMessage]:
    """向模型声明工具装载的变化。

    ``context.tools`` 是运行时可执行的集合；记录中的 system 消息则声明模型可调用
    的集合。每次请求前，两者的差集会写入某条 system 消息的 ``tools_added`` 与
    ``tools_removed``。若待处理消息中已有 system 消息，其工具字段视为意图，
    会被替换为「已提交记录」与「可执行集合」的差值，从而回放时总是精确得到
    ``context.tools``；否则在第一条非 system 待处理消息前插入新的 system 消息。

    Args:
        context: 当前会话上下文，其 ``tools`` 为本回合可执行工具集。
        pending_messages: 本回合待发送的消息列表。

    Returns:
        可能插入或替换 system 消息后的消息列表。
    """
    system_index = -1
    for index in range(len(pending_messages) - 1, -1, -1):
        if getattr(pending_messages[index], "role", None) == "system":
            system_index = index
            break
    pending = pending_messages[system_index] if system_index >= 0 else None
    if pending is not None:
        baseline = [
            with_tool_changes(message, _NO_CHANGES) if index == system_index else message
            for index, message in enumerate(pending_messages)
        ]
    else:
        baseline = pending_messages
    changes = get_tool_state_changes(
        get_current_tools([*context.messages, *baseline]),
        [to_tool_declaration(tool) for tool in (context.tools or [])],
    )
    unchanged = len(changes.tools_added) == 0 and len(changes.tools_removed) == 0

    if pending is not None:
        # 若调用方的消息对象本身已声明无工具变化，则原样保留该对象。
        if unchanged and not (pending.tools_added or pending.tools_removed):
            return pending_messages
        return [
            with_tool_changes(pending, changes) if index == system_index else message
            for index, message in enumerate(baseline)
        ]
    if unchanged:
        return pending_messages
    update = with_tool_changes(SystemMessage(role="system", content="", timestamp=now_ms()), changes)
    insert_index = next(
        (index for index, message in enumerate(pending_messages) if getattr(message, "role", None) != "system"),
        -1,
    )
    index = insert_index if insert_index != -1 else len(pending_messages)
    return [*pending_messages[:index], update, *pending_messages[index:]]


_NO_CHANGES = get_tool_state_changes([], [])


def with_tool_changes(message: SystemMessage, changes: Any) -> SystemMessage:
    """复制 system 消息并把其工具字段替换为 ``changes``。

    空列表表示省略该字段，与 TypeScript 的条件展开一致。

    Args:
        message: 要复制的 system 消息。
        changes: 工具增删变化，含 ``tools_added`` 与 ``tools_removed``。

    Returns:
        工具字段被替换后的 system 消息副本。
    """
    return replace(
        message,
        tools_added=list(changes.tools_added) if changes.tools_added else None,
        tools_removed=list(changes.tools_removed) if changes.tools_removed else None,
    )


async def stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Any,
    emit: AgentEventSink,
    stream_function: StreamFn,
) -> AssistantMessage:
    """从 LLM 流式获取一条 assistant 响应。

    这是 ``AgentMessage`` 被转换为 :data:`pi_ai.types.Message` 供 LLM 使用的地方。

    Args:
        context: 当前会话上下文。
        config: agent 循环配置。
        signal: 取消信号；为 ``None`` 时不响应取消。
        emit: 事件 sink。
        stream_function: 实际发起 stream 请求的函数。

    Returns:
        流式收到的最终 assistant 消息。
    """
    # 若配置了上下文变换则先应用（AgentMessage[] -> AgentMessage[]）
    messages = context.messages
    if config.transform_context is not None:
        messages = await maybe_await(config.transform_context(messages, signal))

    # 转换为 LLM 兼容消息（AgentMessage[] -> Message[]）
    llm_messages = await maybe_await(config.convert_to_llm(messages))

    llm_context = normalize_context(Context(messages=list(llm_messages)))

    # 解析 API key（对会过期的 token 很重要）
    resolved_api_key: str | None = None
    if config.get_api_key is not None:
        resolved_api_key = await maybe_await(config.get_api_key(config.model.provider))
    resolved_api_key = resolved_api_key or config.api_key

    response = await maybe_await(
        stream_function(config.model, llm_context, replace(config, api_key=resolved_api_key, signal=signal))
    )

    # 记录请求的档位，无论实际由哪个 stream 函数响应。
    async def result() -> AssistantMessage:
        """等待最终结果，并补记本次请求实际使用的 thinking level。

        Returns:
            带 ``thinking_level`` 的最终 assistant 消息。
        """
        final = await response.result()
        final.thinking_level = config.reasoning if config.reasoning is not None else "off"
        return final

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            await emit(MessageStartEvent(message=_copy(partial_message)))
        elif event.type in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            if partial_message is not None:
                partial_message = event.partial
                context.messages[len(context.messages) - 1] = partial_message
                await emit(MessageUpdateEvent(message=_copy(partial_message), assistant_message_event=event))
        elif event.type in ("done", "error"):
            final_message = await result()
            if added_partial:
                context.messages[len(context.messages) - 1] = final_message
            else:
                context.messages.append(final_message)
                await emit(MessageStartEvent(message=_copy(final_message)))
            await emit(MessageEndEvent(message=final_message))
            return final_message

    final_message = await result()
    if added_partial:
        context.messages[len(context.messages) - 1] = final_message
    else:
        context.messages.append(final_message)
        await emit(MessageStartEvent(message=_copy(final_message)))
    await emit(MessageEndEvent(message=final_message))
    return final_message


def _copy(message: Any) -> Any:
    """浅拷贝，TypeScript ``{ ...message }`` 展开的移植。

    Args:
        message: 任意消息对象。

    Returns:
        该消息的浅拷贝。
    """
    return copy.copy(message)


async def fail_tool_calls_from_truncated_message(
    tool_calls: list[AgentToolCall],
    emit: AgentEventSink,
) -> "_ExecutedToolCallBatch":
    """把因输出 token 上限被截断的 assistant 消息中的所有 tool call 标记为失败。

    流式 tool call 参数由尽力而为的 JSON 抢救解析器收尾，因此被截断的消息可能
    产出参数「能解析、能通过校验、实则静默不完整」的 tool call。它们都不适合
    执行；逐个报告为错误，让模型重新发起。

    Args:
        tool_calls: 被截断消息中的 tool call 列表。
        emit: 事件 sink。

    Returns:
        全部标记为失败的 tool call 执行批次。
    """
    messages: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await emit(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            )
        )
        finalized = _FinalizedToolCallOutcome(
            tool_call=tool_call,
            result=create_error_tool_result(
                f'Tool call "{tool_call.name}" was not executed: the response hit the output '
                "token limit, so its arguments may be truncated. Re-issue the tool call with "
                "complete arguments."
            ),
            is_error=True,
        )
        await emit_tool_execution_end(finalized, emit)
        tool_result_message = create_tool_result_message(finalized)
        await emit_tool_result_message(tool_result_message, emit)
        messages.append(tool_result_message)
    return _ExecutedToolCallBatch(messages=messages, terminate=False)


async def execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: Any,
    emit: AgentEventSink,
) -> "_ExecutedToolCallBatch":
    """执行 assistant 消息中的 tool call。

    若配置要求串行，或批次中含 ``execution_mode == "sequential"`` 的工具，
    则走串行路径，否则并发执行。

    Args:
        current_context: 当前会话上下文。
        assistant_message: 发起这些 tool call 的 assistant 消息。
        config: agent 循环配置。
        signal: 取消信号；为 ``None`` 时不响应取消。
        emit: 事件 sink。

    Returns:
        本批次 tool call 的执行结果。
    """
    tool_calls = [content for content in assistant_message.content if content.type == "toolCall"]
    has_sequential_tool_call = any(
        (tool := _find_tool(current_context.tools, tool_call.name)) is not None
        and tool.execution_mode == "sequential"
        for tool_call in tool_calls
    )
    if config.tool_execution == "sequential" or has_sequential_tool_call:
        return await execute_tool_calls_sequential(
            current_context, assistant_message, tool_calls, config, signal, emit
        )
    return await execute_tool_calls_parallel(
        current_context, assistant_message, tool_calls, config, signal, emit
    )


def _find_tool(tools: list[AgentTool] | None, name: str) -> AgentTool | None:
    """按名字查找工具。

    Args:
        tools: 候选工具列表，可为 ``None``。
        name: 工具名。

    Returns:
        匹配的 :class:`AgentTool`；未找到时返回 ``None``。
    """
    for tool in tools or []:
        if tool.name == name:
            return tool
    return None


@dataclass
class _ExecutedToolCallBatch:
    """一批 tool call 的执行结果。

    Attributes:
        messages: 每个 tool call 对应的 tool result 消息。
        terminate: 该批次是否要求终止后续循环。
    """

    messages: list[ToolResultMessage] = field(default_factory=list)
    terminate: bool = False


async def execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[AgentToolCall],
    config: AgentLoopConfig,
    signal: Any,
    emit: AgentEventSink,
) -> "_ExecutedToolCallBatch":
    """逐个执行批次中的 tool call。

    某个 tool call 完成后若发现 signal 已中止，则不再继续后续调用。

    Args:
        current_context: 当前会话上下文。
        assistant_message: 发起这些 tool call 的 assistant 消息。
        tool_calls: 要执行的 tool call 列表。
        config: agent 循环配置。
        signal: 取消信号；为 ``None`` 时不响应取消。
        emit: 事件 sink。

    Returns:
        本批次 tool call 的执行结果。
    """
    finalized_calls: list[_FinalizedToolCallOutcome] = []
    messages: list[ToolResultMessage] = []

    for tool_call in tool_calls:
        await emit(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            )
        )

        preparation = await prepare_tool_call(current_context, assistant_message, tool_call, config, signal)
        if preparation.kind == "immediate":
            finalized = _FinalizedToolCallOutcome(
                tool_call=tool_call,
                result=preparation.result,
                is_error=preparation.is_error,
            )
        else:
            executed = await execute_prepared_tool_call(
                preparation, signal, emit_tool_execution_update(tool_call, emit)
            )
            finalized = await finalize_executed_tool_call(
                current_context, assistant_message, preparation, executed, config, signal
            )

        await emit_tool_execution_end(finalized, emit)
        tool_result_message = create_tool_result_message(finalized)
        await emit_tool_result_message(tool_result_message, emit)
        finalized_calls.append(finalized)
        messages.append(tool_result_message)

        if signal is not None and signal.aborted:
            break

    return _ExecutedToolCallBatch(messages=messages, terminate=should_terminate_tool_batch(finalized_calls))


async def execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[AgentToolCall],
    config: AgentLoopConfig,
    signal: Any,
    emit: AgentEventSink,
) -> "_ExecutedToolCallBatch":
    """先依次预检，再并发运行被允许的工具。

    Args:
        current_context: 当前会话上下文。
        assistant_message: 发起这些 tool call 的 assistant 消息。
        tool_calls: 要执行的 tool call 列表。
        config: agent 循环配置。
        signal: 取消信号；为 ``None`` 时不响应取消。
        emit: 事件 sink。

    Returns:
        本批次 tool call 的执行结果，顺序与 ``tool_calls`` 一致。
    """
    finalized_calls: list[Any] = []

    for tool_call in tool_calls:
        await emit(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            )
        )

        preparation = await prepare_tool_call(current_context, assistant_message, tool_call, config, signal)
        if preparation.kind == "immediate":
            finalized = _FinalizedToolCallOutcome(
                tool_call=tool_call,
                result=preparation.result,
                is_error=preparation.is_error,
            )
            await emit_tool_execution_end(finalized, emit)
            finalized_calls.append(finalized)
            if signal is not None and signal.aborted:
                break
            continue

        finalized_calls.append(_deferred_finalize(preparation, current_context, assistant_message, config, signal, emit, tool_call))
        if signal is not None and signal.aborted:
            break

    ordered_finalized_calls = await asyncio.gather(
        *(entry() if callable(entry) else _ready(entry) for entry in finalized_calls)
    )
    messages: list[ToolResultMessage] = []
    for finalized in ordered_finalized_calls:
        tool_result_message = create_tool_result_message(finalized)
        await emit_tool_result_message(tool_result_message, emit)
        messages.append(tool_result_message)

    return _ExecutedToolCallBatch(
        messages=messages,
        terminate=should_terminate_tool_batch(ordered_finalized_calls),
    )


async def _ready(value: Any) -> Any:
    """把已计算完成的值包装成 awaitable，便于与协程一起 ``gather``。

    Args:
        value: 任意已就绪的值。

    Returns:
        原样返回 ``value``。
    """
    return value


def _deferred_finalize(
    preparation: "_PreparedToolCall",
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: Any,
    emit: AgentEventSink,
    tool_call: AgentToolCall,
) -> Callable[[], Any]:
    """把延迟执行的收尾逻辑封装成无参协程工厂。

    并发路径要先完成全部预检，再统一 ``gather`` 真正的执行，故这里返回工厂
    而非协程，避免预检阶段就启动协程。

    Args:
        preparation: 已通过预检的 tool call。
        current_context: 当前会话上下文。
        assistant_message: 发起该 tool call 的 assistant 消息。
        config: agent 循环配置。
        signal: 取消信号；为 ``None`` 时不响应取消。
        emit: 事件 sink。
        tool_call: 该工厂负责的 tool call。

    Returns:
        返回 :class:`_FinalizedToolCallOutcome` 的无参协程工厂。
    """

    async def run() -> _FinalizedToolCallOutcome:
        """执行并收尾该 tool call，同时产出结束事件。

        Returns:
            收尾后的 tool call 结果。
        """
        if signal is not None and signal.aborted:
            finalized = _FinalizedToolCallOutcome(
                tool_call=tool_call,
                result=create_error_tool_result("Operation aborted"),
                is_error=True,
            )
            await emit_tool_execution_end(finalized, emit)
            return finalized
        executed = await execute_prepared_tool_call(
            preparation, signal, emit_tool_execution_update(tool_call, emit)
        )
        finalized = await finalize_executed_tool_call(
            current_context, assistant_message, preparation, executed, config, signal
        )
        await emit_tool_execution_end(finalized, emit)
        return finalized

    return run


@dataclass
class _PreparedToolCall:
    """已通过参数校验、可以执行的 tool call。

    Attributes:
        tool_call: 原始 tool call。
        tool: 解析到的工具。
        args: 校验后的参数。
        kind: 判别标记，固定为 ``"prepared"``。
    """

    tool_call: AgentToolCall
    tool: AgentTool
    args: Any
    kind: str = "prepared"


@dataclass
class _ImmediateToolCallOutcome:
    """无需真正执行即可结算的 tool call 结果。

    Attributes:
        result: 直接产出的结果。
        is_error: 是否为错误结果。
        kind: 判别标记，固定为 ``"immediate"``。
    """

    result: AgentToolResult
    is_error: bool
    kind: str = "immediate"


@dataclass
class _ExecutedToolCallOutcome:
    """工具 ``execute`` 的执行结果。

    Attributes:
        result: 工具返回的结果。
        is_error: 是否为错误结果。
    """

    result: AgentToolResult
    is_error: bool


@dataclass
class _FinalizedToolCallOutcome:
    """收尾后的 tool call 结果。

    Attributes:
        tool_call: 对应的 tool call。
        result: 应用 ``after_tool_call`` 覆盖后的结果。
        is_error: 是否为错误结果。
    """

    tool_call: AgentToolCall = field(default_factory=ToolCall)
    result: AgentToolResult = field(default_factory=AgentToolResult)
    is_error: bool = False


@dataclass
class RunToolCallOptions:
    """:func:`run_tool_call` 的选项。"""

    #: 用于解析本次调用的工具集。
    tools: list[AgentTool] = field(default_factory=list)
    #: 作为发起该调用的消息传给各 hook。
    assistant_message: Any = field(default=None, metadata={"alias": "assistantMessage"})
    #: 作为当前 agent 上下文传给各 hook。
    context: AgentContext = field(default_factory=AgentContext)
    signal: Any = None
    on_update: Callable[[AgentToolResult], Any] | None = field(default=None, metadata={"alias": "onUpdate"})
    before_tool_call: Callable[..., Any] | None = field(default=None, metadata={"alias": "beforeToolCall"})
    after_tool_call: Callable[..., Any] | None = field(default=None, metadata={"alias": "afterToolCall"})


def should_terminate_tool_batch(finalized_calls: list[_FinalizedToolCallOutcome]) -> bool:
    """批次内每个收尾结果的 ``terminate`` 是否都为 true。

    Args:
        finalized_calls: 已收尾的 tool call 结果列表。

    Returns:
        列表非空且每项都要求终止时返回 ``True``，否则返回 ``False``。
    """
    return len(finalized_calls) > 0 and all(finalized.result.terminate is True for finalized in finalized_calls)


def prepare_tool_call_arguments(tool: AgentTool, tool_call: AgentToolCall) -> AgentToolCall:
    """在校验前应用工具的可选 ``prepare_arguments`` 垫片。

    Args:
        tool: 目标工具。
        tool_call: 原始 tool call。

    Returns:
        应用垫片后的 tool call；未定义该钩子或参数对象未变时原样返回。
    """
    if tool.prepare_arguments is None:
        return tool_call
    prepared_arguments = tool.prepare_arguments(tool_call.arguments)
    if prepared_arguments is tool_call.arguments:
        return tool_call
    return replace(tool_call, arguments=prepared_arguments)


async def prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: AgentToolCall,
    config: Any,
    signal: Any,
    tools: list[AgentTool] | None = None,
) -> _PreparedToolCall | _ImmediateToolCallOutcome:
    """解析、校验并预检一个 tool call；工具故障绝不抛出。

    Args:
        current_context: 当前会话上下文。
        assistant_message: 发起该 tool call 的 assistant 消息。
        tool_call: 要预检的 tool call。
        config: 含 ``before_tool_call`` 钩子的配置对象。
        signal: 取消信号；为 ``None`` 时不响应取消。
        tools: 可选的工具集覆盖；为 ``None`` 时使用 ``current_context.tools``。

    Returns:
        可执行的 :class:`_PreparedToolCall`，或立即失败的
        :class:`_ImmediateToolCallOutcome`。
    """
    resolved_tools = current_context.tools if tools is None else tools
    tool = _find_tool(resolved_tools, tool_call.name)
    if tool is None:
        return _ImmediateToolCallOutcome(
            result=create_error_tool_result(f"Tool {tool_call.name} not found"),
            is_error=True,
        )

    try:
        prepared_tool_call = prepare_tool_call_arguments(tool, tool_call)
        validated_args = validate_tool_arguments(tool, prepared_tool_call)
        if config.before_tool_call is not None:
            before_result = _coerce_optional(
                BeforeToolCallResult,
                await maybe_await(
                    config.before_tool_call(
                        _before_context(assistant_message, tool_call, validated_args, current_context),
                        signal,
                    )
                ),
            )
            if signal is not None and signal.aborted:
                return _ImmediateToolCallOutcome(
                    result=create_error_tool_result("Operation aborted"),
                    is_error=True,
                )
            if before_result is not None and before_result.block:
                result = create_error_tool_result(before_result.reason or "Tool execution was blocked")
                if before_result.terminate is True:
                    result.terminate = True
                return _ImmediateToolCallOutcome(result=result, is_error=True)
        if signal is not None and signal.aborted:
            return _ImmediateToolCallOutcome(
                result=create_error_tool_result("Operation aborted"),
                is_error=True,
            )
        return _PreparedToolCall(tool_call=tool_call, tool=tool, args=validated_args)
    except Exception as error:  # noqa: BLE001 - 工具故障转为错误结果
        return _ImmediateToolCallOutcome(
            result=create_error_tool_result(str(error)),
            is_error=True,
        )


def _before_context(
    assistant_message: AssistantMessage,
    tool_call: AgentToolCall,
    args: Any,
    context: AgentContext,
) -> BeforeToolCallContext:
    """构建传给 ``before_tool_call`` 的上下文。

    Args:
        assistant_message: 发起该 tool call 的 assistant 消息。
        tool_call: 当前 tool call。
        args: 校验后的参数。
        context: 当前会话上下文。

    Returns:
        供钩子使用的 :class:`BeforeToolCallContext`。
    """
    return BeforeToolCallContext(
        assistant_message=assistant_message,
        tool_call=tool_call,
        args=args,
        context=context,
    )


def _after_context(
    assistant_message: AssistantMessage,
    tool_call: AgentToolCall,
    args: Any,
    result: AgentToolResult,
    is_error: bool,
    context: AgentContext,
) -> AfterToolCallContext:
    """构建传给 ``after_tool_call`` 的上下文。

    Args:
        assistant_message: 发起该 tool call 的 assistant 消息。
        tool_call: 当前 tool call。
        args: 校验后的参数。
        result: 工具执行结果。
        is_error: 结果是否为错误。
        context: 当前会话上下文。

    Returns:
        供钩子使用的 :class:`AfterToolCallContext`。
    """
    return AfterToolCallContext(
        assistant_message=assistant_message,
        tool_call=tool_call,
        args=args,
        result=result,
        is_error=is_error,
        context=context,
    )


def emit_tool_execution_update(tool_call: AgentToolCall, emit: AgentEventSink) -> Callable[[AgentToolResult], Any]:
    """构建该次 tool call 专用的部分结果回调。

    Args:
        tool_call: 归属的 tool call。
        emit: 事件 sink。

    Returns:
        接收部分结果并产出 ``tool_execution_update`` 事件的回调。
    """

    def sink(partial_result: AgentToolResult) -> Any:
        """把部分结果包装为 ``tool_execution_update`` 事件。

        Args:
            partial_result: 工具执行过程中产出的部分结果。

        Returns:
            ``emit`` 的返回值。
        """
        return emit(
            ToolExecutionUpdateEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
                partial_result=partial_result,
            )
        )

    return sink


async def run_tool_call(tool_call: AgentToolCall, options: RunToolCallOptions) -> AgentToolCallOutcome:
    """按与模型发起调用完全相同的步骤运行一个 tool call。

    参数准备、schema 校验、``before_tool_call``、执行与 ``after_tool_call``
    一应俱全。不产出任何事件，也不追加任何消息。内部会调用其他工具的工具
    使用本函数，使各 hook（例如权限检查）同样作用于这些调用。

    工具故障绝不拒绝：未知工具、校验错误、被拦截的调用与抛出的异常
    都会以 ``is_error=True`` 返回。

    Args:
        tool_call: 要运行的 tool call。
        options: 运行选项，含工具集、上下文与各 hook。

    Returns:
        本次调用的最终结果。
    """
    assistant_message = options.assistant_message
    context = options.context
    signal = options.signal
    preparation = await prepare_tool_call(
        context, assistant_message, tool_call, options, signal, list(options.tools)
    )
    if preparation.kind == "immediate":
        return AgentToolCallOutcome(tool_call=tool_call, result=preparation.result, is_error=preparation.is_error)
    on_update = options.on_update if options.on_update is not None else (lambda _partial: None)
    executed = await execute_prepared_tool_call(preparation, signal, on_update)
    finalized = await finalize_executed_tool_call(
        context, assistant_message, preparation, executed, options, signal
    )
    return AgentToolCallOutcome(
        tool_call=finalized.tool_call,
        result=finalized.result,
        is_error=finalized.is_error,
    )


def _positional_arity(func: Callable[..., Any]) -> int | None:
    """统计 ``func`` 声明的位置参数个数；若含 ``*args`` 则返回 ``None``。

    TypeScript 工具可以只声明完整签名 ``(toolCallId, params, signal, onUpdate)``
    的前几个参数并忽略其余；Python 可调用对象做不到。循环通过只传入可调用对象
    能接受的参数个数，来对齐 TypeScript 的宽容行为。

    Args:
        func: 待检查的可调用对象。

    Returns:
        位置参数个数；若签名含 ``*args`` 或无法获取签名则返回 ``None``。
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - 内建函数没有可用签名
        return None
    count = 0
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return None
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            count += 1
    return count


def _call_tool_execute(
    execute: Callable[..., Any],
    tool_call_id: str,
    args: Any,
    signal: Any,
    on_update: Callable[[AgentToolResult], Any],
) -> Any:
    """按工具自身声明的参数个数调用其 ``execute``。

    Args:
        execute: 工具的 ``execute`` 可调用对象。
        tool_call_id: tool call 的 id。
        args: 已完成校验的参数。
        signal: 取消信号。
        on_update: 部分结果回调。

    Returns:
        ``execute`` 的返回值。
    """
    arity = _positional_arity(execute)
    arguments = (tool_call_id, args, signal, on_update)
    if arity is None:
        return execute(*arguments)
    return execute(*arguments[: min(arity, len(arguments))])


async def execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    signal: Any,
    on_update: Callable[[AgentToolResult], Any],
) -> _ExecutedToolCallOutcome:
    """执行一个已准备的 tool call，收集更新事件直至其结算。

    Args:
        prepared: 已通过预检的 tool call。
        signal: 取消信号；为 ``None`` 时不响应取消。
        on_update: 部分结果回调。

    Returns:
        执行结果，含是否出错的标记。
    """
    update_events: list[asyncio.Future[Any]] = []
    accepting_updates = True

    def update_sink(partial_result: AgentToolResult) -> None:
        """收集工具产出的部分结果，交由 ``on_update`` 异步转发。

        Args:
            partial_result: 工具执行过程中产出的部分结果。
        """
        nonlocal accepting_updates
        if not accepting_updates:
            return
        update_events.append(asyncio.ensure_future(maybe_await(on_update(partial_result))))

    try:
        tool = prepared.tool
        if tool.execute is None:
            raise ValueError(f'Tool "{tool.name}" has no execute function')
        result = await maybe_await(
            _call_tool_execute(tool.execute, prepared.tool_call.id, prepared.args, signal, update_sink)
        )
        accepting_updates = False
        if update_events:
            await asyncio.gather(*update_events)
        return _ExecutedToolCallOutcome(result=result, is_error=result.is_error is True)
    except Exception as error:  # noqa: BLE001 - 工具故障转为错误结果
        accepting_updates = False
        if update_events:
            await asyncio.gather(*update_events)
        return _ExecutedToolCallOutcome(result=create_error_tool_result(str(error)), is_error=True)
    finally:
        accepting_updates = False


async def finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    executed: _ExecutedToolCallOutcome,
    config: Any,
    signal: Any,
) -> _FinalizedToolCallOutcome:
    """对执行结果应用可选的 ``after_tool_call`` 覆盖。

    Args:
        current_context: 当前会话上下文。
        assistant_message: 发起该 tool call 的 assistant 消息。
        prepared: 已通过预检的 tool call。
        executed: 工具执行结果。
        config: 含 ``after_tool_call`` 钩子的配置对象。
        signal: 取消信号；为 ``None`` 时不响应取消。

    Returns:
        应用覆盖后的收尾结果。
    """
    result = executed.result
    is_error = executed.is_error

    if config.after_tool_call is not None:
        try:
            after_result = _coerce_optional(
                AfterToolCallResult,
                await maybe_await(
                    config.after_tool_call(
                        _after_context(
                            assistant_message,
                            prepared.tool_call,
                            prepared.args,
                            result,
                            is_error,
                            current_context,
                        ),
                        signal,
                    )
                ),
            )
            if after_result is not None:
                # 未随 content 一并替换的结构化内容可能已与 content 不再匹配。
                if after_result.structured_content is not None:
                    structured_content = after_result.structured_content
                elif after_result.content is not None:
                    structured_content = None
                else:
                    structured_content = result.structured_content
                result = replace(
                    result,
                    content=after_result.content if after_result.content is not None else result.content,
                    details=after_result.details if after_result.details is not None else result.details,
                    usage=after_result.usage if after_result.usage is not None else result.usage,
                    terminate=after_result.terminate if after_result.terminate is not None else result.terminate,
                    structured_content=structured_content,
                )
                is_error = after_result.is_error if after_result.is_error is not None else is_error
        except Exception as error:  # noqa: BLE001 - hook 故障转为错误结果
            result = create_error_tool_result(str(error))
            is_error = True

    return _FinalizedToolCallOutcome(tool_call=prepared.tool_call, result=result, is_error=is_error)


def create_error_tool_result(message: str) -> AgentToolResult:
    """文本内容、空 details、且不带错误标志的错误结果。

    ``is_error`` 由调用方单独决定，与 ``createErrorToolResult`` 保持一致。

    Args:
        message: 错误文本。

    Returns:
        仅含文本内容与空 ``details`` 的 :class:`AgentToolResult`。
    """
    return AgentToolResult(content=[TextContent(text=message)], details={})


async def emit_tool_execution_end(finalized: _FinalizedToolCallOutcome, emit: AgentEventSink) -> None:
    """为已收尾的调用产出 ``tool_execution_end``。

    Args:
        finalized: 已收尾的 tool call 结果。
        emit: 事件 sink。
    """
    await emit(
        ToolExecutionEndEvent(
            tool_call_id=finalized.tool_call.id,
            tool_name=finalized.tool_call.name,
            result=finalized.result,
            is_error=finalized.is_error,
        )
    )


def create_tool_result_message(finalized: _FinalizedToolCallOutcome) -> ToolResultMessage:
    """为已收尾的 tool call 构建会话记录产物。

    Args:
        finalized: 已收尾的 tool call 结果。

    Returns:
        对应的 :class:`ToolResultMessage`。
    """
    return ToolResultMessage(
        role="toolResult",
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        # 无类型工具（Python 扩展）可能返回没有 content 的结果；此处归一化，
        # 避免 null 进入会话历史或 provider 请求体。
        content=finalized.result.content if finalized.result.content else [],
        details=finalized.result.details,
        usage=finalized.result.usage,
        is_error=finalized.is_error,
        timestamp=now_ms(),
    )


async def emit_tool_result_message(tool_result_message: ToolResultMessage, emit: AgentEventSink) -> None:
    """为 tool result 产出 ``message_start``/``message_end`` 事件对。

    Args:
        tool_result_message: 要产出的 tool result 消息。
        emit: 事件 sink。
    """
    await emit(MessageStartEvent(message=tool_result_message))
    await emit(MessageEndEvent(message=tool_result_message))
