"""围绕底层 agent 循环的有状态 :class:`Agent` 封装。

移植自 ``packages/agent/src/agent.ts``。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from pi_ai.types import (
    AssistantMessage,
    Model,
    ModelCost,
    TextContent,
    ThinkingBudgets,
    Transport,
    Usage,
    UserMessage,
)
from pi_ai.utils.abort import AbortController, AbortSignal
from pi_ai.utils.async_utils import maybe_await
from pi_ai.utils.transcript import (
    create_initial_system_message,
    get_current_system_message,
    to_tool_declaration,
)

from .agent_loop import now_ms, run_agent_loop, run_agent_loop_continue
from .stream_fn import get_default_stream_fn
from .types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentInitialState,
    AgentLoopConfig,
    AgentState,
    MessageEndEvent,
    MessageStartEvent,
    QueueMode,
    StreamFn,
    ToolExecutionMode,
    TurnEndEvent,
)

__all__ = ["Agent", "AgentInitialState", "AgentOptions", "QueueMode"]


def default_convert_to_llm(messages: list[Any]) -> list[Any]:
    """只保留所有 LLM provider 都能理解的消息角色。

    Args:
        messages: 待过滤的消息列表。

    Returns:
        仅含 system、user、assistant、toolResult 角色的消息列表。
    """
    return [
        message
        for message in messages
        if getattr(message, "role", None) in ("system", "user", "assistant", "toolResult")
    ]


def _empty_usage() -> Usage:
    """``agent.ts`` 中 ``EMPTY_USAGE`` 常量的对应物，每条失败消息都重新构建。"""
    return Usage(cost=ModelCost())


def _default_model() -> Model:
    """``agent.ts`` 中 ``DEFAULT_MODEL`` 占位符的对应物。"""
    return Model(
        id="unknown",
        name="unknown",
        api="unknown",
        provider="unknown",
        base_url="",
        reasoning=False,
        input=[],
        cost=ModelCost(),
        context_window=0,
        max_tokens=0,
    )


def create_mutable_agent_state(initial_state: AgentInitialState | None = None) -> AgentState:
    """构建 :class:`Agent` 持有的可变状态。

    Args:
        initial_state: 初始状态；为 ``None`` 时全部使用默认值。

    Returns:
        新建的可变 agent 状态。
    """
    tools = list(initial_state.tools) if initial_state is not None and initial_state.tools is not None else []
    messages = (
        list(initial_state.messages) if initial_state is not None and initial_state.messages is not None else []
    )
    initial_message = create_initial_system_message(
        initial_state.system_prompt if initial_state is not None else None,
        [to_tool_declaration(tool) for tool in tools],
    )
    if (not messages or getattr(messages[0], "role", None) != "system") and initial_message is not None:
        messages.insert(0, initial_message)

    model = _default_model()
    if initial_state is not None and initial_state.model is not None:
        model = initial_state.model
    thinking_level: Any = "off"
    if initial_state is not None and initial_state.thinking_level is not None:
        thinking_level = initial_state.thinking_level
    return AgentState(messages=messages, tools=tools, model=model, thinking_level=thinking_level)



class PendingMessageQueue:
    """steering/follow-up 消息队列，支持可配置的出队模式。

    Attributes:
        mode: 出队模式；``"all"`` 一次取完全部消息，``"one-at-a-time"`` 每次只取
            最旧的一条。
    """

    def __init__(self, mode: QueueMode | str) -> None:
        """初始化队列。

        Args:
            mode: 出队模式。
        """
        self._messages: list[Any] = []
        self.mode: QueueMode | str = mode

    def enqueue(self, message: Any) -> None:
        """将 ``message`` 追加到队列末尾。

        Args:
            message: 要排队的消息。
        """
        self._messages.append(message)

    def has_items(self) -> bool:
        """队列中是否至少有一条消息。"""
        return len(self._messages) > 0

    def peek(self) -> list[Any]:
        """预览下次 drain 将取出的消息，不消费它们。"""
        if self.mode == "all":
            return list(self._messages)
        first = self._messages[0] if self._messages else None
        return [first] if first is not None else []

    def drain(self) -> list[Any]:
        """取出并返回当前模式选中的消息。"""
        drained = self.peek()
        self._messages = self._messages[len(drained) :]
        return drained

    def clear(self) -> None:
        """清空所有排队消息。"""
        self._messages = []


@dataclass
class AgentOptions:
    """构造 :class:`Agent` 的选项。

    Attributes:
        initial_state: agent 的初始状态。
        convert_to_llm: 每次 LLM 调用前的消息转换回调。
        transform_context: ``convert_to_llm`` 之前应用的上下文变换回调。
        stream_fn: provider stream 函数；未提供时使用默认 stream 函数。
        get_api_key: 动态解析 API key 的回调。
        on_payload: 请求发出前观察或改写 payload 的回调。
        on_response: 收到 provider 响应后调用的回调。
        on_provider_stream_event: 收到 provider 原始 stream 事件时调用的回调。
        before_tool_call: 工具执行前的拦截回调。
        after_tool_call: 工具执行后的结果覆盖回调。
        finish_turn: assistant 回合完成后的回合完成回调。
        prepare_request: 每次会话型 provider 请求之前的回调。
        prepare_next_turn: ``turn_end`` 之后、下一回合开始之前的回调。
        prepare_next_turn_with_context: 额外接收上下文与 abort signal 的
            ``prepare_next_turn`` 变体。
        steering_mode: steering 消息的出队模式。
        follow_up_mode: follow-up 消息的出队模式。
        session_id: 会话标识，转发给支持缓存的 provider 后端。
        thinking_budgets: 各 thinking 档位对应的 token 预算。
        transport: 转发给 stream 函数的首选传输方式。
        max_retry_delay_ms: provider 请求重试延迟的上限（毫秒）。
        tool_execution: 含多个 tool call 的 assistant 消息的工具执行策略。
    """

    initial_state: AgentInitialState | None = field(default=None, metadata={"alias": "initialState"})
    convert_to_llm: Callable[..., Any] | None = field(default=None, metadata={"alias": "convertToLlm"})
    transform_context: Callable[..., Any] | None = field(default=None, metadata={"alias": "transformContext"})
    stream_fn: StreamFn | None = field(default=None, metadata={"alias": "streamFn"})
    get_api_key: Callable[..., Any] | None = field(default=None, metadata={"alias": "getApiKey"})
    on_payload: Callable[..., Any] | None = field(default=None, metadata={"alias": "onPayload"})
    on_response: Callable[..., Any] | None = field(default=None, metadata={"alias": "onResponse"})
    on_provider_stream_event: Callable[..., Any] | None = field(
        default=None, metadata={"alias": "onProviderStreamEvent"}
    )
    before_tool_call: Callable[..., Any] | None = field(default=None, metadata={"alias": "beforeToolCall"})
    after_tool_call: Callable[..., Any] | None = field(default=None, metadata={"alias": "afterToolCall"})
    finish_turn: Callable[..., Any] | None = field(default=None, metadata={"alias": "finishTurn"})
    prepare_request: Callable[..., Any] | None = field(default=None, metadata={"alias": "prepareRequest"})
    prepare_next_turn: Callable[..., Any] | None = field(default=None, metadata={"alias": "prepareNextTurn"})
    prepare_next_turn_with_context: Callable[..., Any] | None = field(
        default=None, metadata={"alias": "prepareNextTurnWithContext"}
    )
    steering_mode: QueueMode | str | None = field(default=None, metadata={"alias": "steeringMode"})
    follow_up_mode: QueueMode | str | None = field(default=None, metadata={"alias": "followUpMode"})
    session_id: str | None = field(default=None, metadata={"alias": "sessionId"})
    thinking_budgets: ThinkingBudgets | None = field(default=None, metadata={"alias": "thinkingBudgets"})
    transport: Transport | str | None = None
    max_retry_delay_ms: int | None = field(default=None, metadata={"alias": "maxRetryDelayMs"})
    tool_execution: ToolExecutionMode | str | None = field(default=None, metadata={"alias": "toolExecution"})


@dataclass
class _PromptRunOptions:
    """一次 prompt 驱动运行的选项。

    Attributes:
        skip_initial_steering_poll: 为真时跳过本轮的首次 steering 队列轮询。
    """

    skip_initial_steering_poll: bool = False


@dataclass
class _ActiveRun:
    """当前正在运行的 prompt 或续跑的状态。

    Attributes:
        promise: 运行结束时结算的 future，``wait_for_idle`` 会 await 它。
        abort_controller: 本次运行的 abort 控制器。
    """

    promise: asyncio.Future[None]
    abort_controller: AbortController


class Agent:
    """底层 agent 循环的有状态封装。

    ``Agent`` 持有当前会话记录、派发生命周期事件、执行工具，
    并提供 steering 与 follow-up 消息的排队 API。

    Attributes:
        convert_to_llm: 每次 LLM 调用前的消息转换回调。
        transform_context: ``convert_to_llm`` 之前应用的上下文变换回调。
        stream_function: 实际调用的 provider stream 函数。
        get_api_key: 动态解析 API key 的回调。
        on_payload: 请求发出前观察或改写 payload 的回调。
        on_response: 收到 provider 响应后调用的回调。
        on_provider_stream_event: 收到 provider 原始 stream 事件时调用的回调。
        before_tool_call: 工具执行前的拦截回调。
        after_tool_call: 工具执行后的结果覆盖回调。
        finish_turn: assistant 回合完成后的回合完成回调。
        prepare_request: 每次会话型 provider 请求之前的回调。
        prepare_next_turn: ``turn_end`` 之后、下一回合开始之前的回调。
        prepare_next_turn_with_context: 额外接收上下文与 abort signal 的
            ``prepare_next_turn`` 变体。
        session_id: 会话标识，转发给支持缓存的 provider 后端。
        thinking_budgets: 各 thinking 档位对应的 token 预算。
        transport: 转发给 stream 函数的首选传输方式。
        max_retry_delay_ms: provider 请求重试延迟的上限（毫秒）。
        tool_execution: 含多个 tool call 的 assistant 消息的工具执行策略。
    """

    def __init__(
        self,
        options: AgentOptions | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        """初始化 agent。

        Args:
            options: 构造选项；传入普通 ``Mapping`` 时会解包为 :class:`AgentOptions`，
                传入 ``None`` 时使用默认选项。
            **overrides: 覆盖或补充 :class:`AgentOptions` 字段的关键字参数。
        """
        if options is None:
            runtime = AgentOptions()
        elif isinstance(options, Mapping):
            runtime = AgentOptions(**options)
        else:
            runtime = options
        if overrides:
            runtime = replace(runtime, **overrides)

        self._state: AgentState = create_mutable_agent_state(runtime.initial_state)
        self._listeners: list[Callable[[AgentEvent, AbortSignal], Any]] = []
        self._steering_queue = PendingMessageQueue(runtime.steering_mode or "one-at-a-time")
        self._follow_up_queue = PendingMessageQueue(runtime.follow_up_mode or "one-at-a-time")
        self._active_run: _ActiveRun | None = None

        self.convert_to_llm: Callable[..., Any] = runtime.convert_to_llm or default_convert_to_llm
        self.transform_context: Callable[..., Any] | None = runtime.transform_context
        self.stream_function: StreamFn = (
            runtime.stream_fn if runtime.stream_fn is not None else get_default_stream_fn()
        )
        self.get_api_key: Callable[..., Any] | None = runtime.get_api_key
        self.on_payload: Callable[..., Any] | None = runtime.on_payload
        self.on_response: Callable[..., Any] | None = runtime.on_response
        self.on_provider_stream_event: Callable[..., Any] | None = runtime.on_provider_stream_event
        self.before_tool_call: Callable[..., Any] | None = runtime.before_tool_call
        self.after_tool_call: Callable[..., Any] | None = runtime.after_tool_call
        self.finish_turn: Callable[..., Any] | None = runtime.finish_turn
        self.prepare_request: Callable[..., Any] | None = runtime.prepare_request
        self.prepare_next_turn: Callable[..., Any] | None = runtime.prepare_next_turn
        self.prepare_next_turn_with_context: Callable[..., Any] | None = runtime.prepare_next_turn_with_context

        #: 会话标识，转发给支持缓存的 provider 后端。
        self.session_id: str | None = runtime.session_id
        #: 可选的按档位 thinking token 预算，转发给 stream 函数。
        self.thinking_budgets: ThinkingBudgets | None = runtime.thinking_budgets
        #: 首选传输方式，转发给 stream 函数。
        self.transport: Transport | str = runtime.transport if runtime.transport is not None else "auto"
        #: provider 请求重试延迟的上限（毫秒）。
        self.max_retry_delay_ms: int | None = runtime.max_retry_delay_ms
        #: 包含多个 tool call 的 assistant 消息的工具执行策略。
        self.tool_execution: ToolExecutionMode | str = (
            runtime.tool_execution if runtime.tool_execution is not None else "parallel"
        )

    # -- 状态 ----------------------------------------------------------------

    @property
    def state(self) -> AgentState:
        """当前 agent 状态。

        返回内部状态的直接引用。调用方不应直接修改 messages、tools 或其他列表字段，
        应使用相应的方法（steer、follow_up 等）来修改状态。

        Returns:
            内部 :class:`~pi_agent.types.AgentState` 的直接引用。
        """
        return self._state

    @property
    def steering_mode(self) -> QueueMode | str:
        """排队 steering 消息的出队方式。

        Returns:
            当前出队模式。
        """
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode | str) -> None:
        """设置排队 steering 消息的出队方式。

        Args:
            mode: 新的出队模式。
        """
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode | str:
        """排队 follow-up 消息的出队方式。

        Returns:
            当前出队模式。
        """
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode | str) -> None:
        """设置排队 follow-up 消息的出队方式。

        Args:
            mode: 新的出队模式。
        """
        self._follow_up_queue.mode = mode

    # -- 事件 ---------------------------------------------------------------

    def subscribe(self, listener: Callable[[AgentEvent, AbortSignal], Any]) -> Callable[[], None]:
        """订阅 agent 生命周期事件。

        监听器的返回值按订阅顺序 await，其结算计入当前运行的完成判定。
        监听器同时会收到当前运行的 abort signal。

        ``agent_end`` 是一次运行派发的最后一个事件，但在该事件上被 await 的
        所有监听器结算之前，agent 不会进入空闲状态。

        Args:
            listener: 事件监听器，接收事件与当前运行的 abort signal。

        Returns:
            取消本次订阅的无参函数。
        """
        if listener not in self._listeners:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            """取消本次订阅。"""
            self.unsubscribe(listener)

        return unsubscribe

    def unsubscribe(self, listener: Callable[[AgentEvent, AbortSignal], Any]) -> None:
        """移除先前传给 :meth:`subscribe` 的监听器。

        TypeScript 的 ``subscribe`` 返回同一个闭包；本方法是为偏好具名移除的
        调用方提供的 Python 对应物。

        Args:
            listener: 要移除的监听器。
        """
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    # -- 队列 ---------------------------------------------------------------

    def steer(self, message: Any) -> None:
        """排队一条消息，在当前 assistant 回合结束后注入。

        Args:
            message: 要排队注入的消息。
        """
        self._steering_queue.enqueue(message)

    def follow_up(self, message: Any) -> None:
        """排队一条消息，仅在 agent 即将停止时才运行。

        Args:
            message: 要排队的 follow-up 消息。
        """
        self._follow_up_queue.enqueue(message)

    def clear_steering_queue(self) -> None:
        """清空所有排队的 steering 消息。"""
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        """清空所有排队的 follow-up 消息。"""
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        """清空所有排队的 steering 与 follow-up 消息。"""
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        """两个队列中是否仍有待处理消息。"""
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    def peek_queued_messages(self) -> list[Any]:
        """预览下次运行将消费的消息，不实际取出。"""
        steering = self._steering_queue.peek()
        return steering if len(steering) > 0 else self._follow_up_queue.peek()

    # -- 生命周期 ------------------------------------------------------------

    @property
    def signal(self) -> AbortSignal | None:
        """当前运行的 abort signal（若存在）。

        Returns:
            当前运行的 :class:`AbortSignal`；没有活动运行时返回 ``None``。
        """
        if self._active_run is None:
            return None
        return self._active_run.abort_controller.signal

    def abort(self) -> None:
        """中止当前运行（若有）。"""
        if self._active_run is not None:
            self._active_run.abort_controller.abort()

    async def wait_for_idle(self) -> None:
        """等待当前运行及所有被 await 的事件监听器结束后返回。

        会在 ``agent_end`` 监听器结算之后才返回。
        """
        run = self._active_run
        if run is not None:
            await run.promise

    def reset(self) -> None:
        """清空会话状态与队列，但保留回放得到的 prompt/tool 基线。"""
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before resetting.")

        baseline = get_current_system_message(self._state.messages)
        self._state.messages = [baseline] if baseline is not None else []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._state.error_message = None
        self.clear_follow_up_queue()
        self.clear_steering_queue()

    # -- 提示 ------------------------------------------------------------

    async def prompt(
        self,
        message: str | Any | list[Any],
    ) -> None:
        """发起新的 prompt，支持文本、单条消息或消息列表。

        Args:
            message: 文本、单条消息或消息列表。

        Raises:
            RuntimeError: 已有 prompt 或续跑正在进行时。
        """
        if self._active_run is not None:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or followUp() to queue messages, "
                "or wait for completion."
            )
        messages = self._normalize_prompt_input(message)
        await self._run_prompt_messages(messages)

    async def continue_(self) -> None:
        """从当前会话记录继续。

        最后一条消息必须是 user 或 toolResult 消息。
        """
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")

        messages = self._state.messages
        last_message = messages[-1] if messages else None
        if last_message is None or all(getattr(message, "role", None) == "system" for message in messages):
            raise RuntimeError("No messages to continue from")

        if getattr(last_message, "role", None) == "assistant":
            queued_steering = self._steering_queue.drain()
            if len(queued_steering) > 0:
                await self._run_prompt_messages(
                    queued_steering, _PromptRunOptions(skip_initial_steering_poll=True)
                )
                return

            queued_follow_ups = self._follow_up_queue.drain()
            if len(queued_follow_ups) > 0:
                await self._run_prompt_messages(queued_follow_ups)
                return

            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_continuation()

    def _normalize_prompt_input(
        self,
        message: str | Any | list[Any],
    ) -> list[Any]:
        """把 prompt 输入规范化为消息列表。

        Args:
            message: 文本、单条消息或消息列表。

        Returns:
            规范化后的消息列表。
        """
        if isinstance(message, list):
            return message

        if not isinstance(message, str):
            return [message]

        content: list[Any] = [TextContent(text=message)]
        return [UserMessage(role="user", content=content, timestamp=now_ms())]

    async def _run_prompt_messages(
        self,
        messages: list[Any],
        options: _PromptRunOptions | None = None,
    ) -> None:
        """以 prompt 语义运行一轮 agent 循环。

        Args:
            messages: 作为本轮输入的消息。
            options: 本次运行的选项；为 ``None`` 时使用默认值。
        """

        async def executor(signal: AbortSignal) -> None:
            """驱动一次 prompt 循环。

            Args:
                signal: 本次运行的 abort signal。
            """
            await run_agent_loop(
                messages,
                self._create_context_snapshot(),
                self._create_loop_config(options),
                lambda event: self._process_events(event),
                signal,
                self.stream_function,
            )

        await self._run_with_lifecycle(executor)

    async def _run_continuation(self) -> None:
        """从当前会话记录继续驱动循环。"""

        async def executor(signal: AbortSignal) -> None:
            """驱动一次续跑循环。

            Args:
                signal: 本次运行的 abort signal。
            """
            await run_agent_loop_continue(
                self._create_context_snapshot(),
                self._create_loop_config(),
                lambda event: self._process_events(event),
                signal,
                self.stream_function,
            )

        await self._run_with_lifecycle(executor)

    def _create_context_snapshot(self) -> AgentContext:
        """创建传给底层循环的上下文快照。

        Returns:
            包含当前消息与工具副本的上下文。
        """
        return AgentContext(messages=list(self._state.messages), tools=list(self._state.tools))

    def _create_loop_config(self, options: _PromptRunOptions | None = None) -> AgentLoopConfig:
        """组装传给底层循环的配置。

        Args:
            options: 一次 prompt 驱动运行的选项；为 ``None`` 时使用默认值。

        Returns:
            本次运行使用的 :class:`~pi_agent.types.AgentLoopConfig`。
        """
        skip_initial_steering_poll = bool(options is not None and options.skip_initial_steering_poll)
        thinking_level = self._state.thinking_level
        prepare_next_turn = None
        if self.prepare_next_turn_with_context is not None or self.prepare_next_turn is not None:

            def prepare_next_turn(context: Any) -> Any:
                """调用调用方注册的下一回合准备回调。

                Args:
                    context: 回合完成上下文。

                Returns:
                    回调返回的下一回合更新；未注册时返回 ``None``。
                """
                if self.prepare_next_turn_with_context is not None:
                    return self.prepare_next_turn_with_context(context, self.signal)
                return self.prepare_next_turn(self.signal)

        def get_steering_messages() -> list[Any]:
            """取出待注入的 steering 消息；首次轮询可能被跳过。"""
            nonlocal skip_initial_steering_poll
            if skip_initial_steering_poll:
                skip_initial_steering_poll = False
                return []
            return self._steering_queue.drain()

        return AgentLoopConfig(
            model=self._state.model,
            reasoning=None if thinking_level == "off" else thinking_level,
            session_id=self.session_id,
            on_payload=self.on_payload,
            on_response=self.on_response,
            on_provider_stream_event=self.on_provider_stream_event,
            transport=self.transport,
            thinking_budgets=self.thinking_budgets,
            max_retry_delay_ms=self.max_retry_delay_ms,
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            finish_turn=self.finish_turn,
            prepare_request=self.prepare_request,
            prepare_next_turn=prepare_next_turn,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=lambda: self._follow_up_queue.drain(),
        )

    async def _run_with_lifecycle(self, executor: Callable[[AbortSignal], Any]) -> None:
        """在完整生命周期内执行传入的 executor。

        负责建立活动运行、设置流式标志，并把失败交给
        :meth:`_handle_run_failure` 处理。

        Args:
            executor: 接收 abort signal、真正驱动循环的异步可调用对象。

        Raises:
            RuntimeError: 已有运行正在进行时。
        """
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing.")

        abort_controller = AbortController()
        promise: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._active_run = _ActiveRun(promise=promise, abort_controller=abort_controller)

        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        try:
            await executor(abort_controller.signal)
        except Exception as error:  # noqa: BLE001 - 失败统一转为 assistant 错误消息
            await self._handle_run_failure(error, abort_controller.signal.aborted)
        finally:
            self._finish_run()

    async def _handle_run_failure(self, error: Any, aborted: bool) -> None:
        """把运行失败转换为错误 assistant 消息并派发。

        Args:
            error: 捕获到的异常。
            aborted: 失败是否由 abort 引起；为真时停止原因记为 ``"aborted"``。
        """
        model = self._state.model
        failure_message = AssistantMessage(
            role="assistant",
            content=[TextContent(text="")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=_empty_usage(),
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
            timestamp=now_ms(),
        )
        await self._process_events(MessageStartEvent(message=failure_message))
        await self._process_events(MessageEndEvent(message=failure_message))
        await self._process_events(TurnEndEvent(message=failure_message, tool_results=[]))
        await self._process_events(AgentEndEvent(messages=[failure_message]))

    def _finish_run(self) -> None:
        """清理运行时状态并结算当前运行的 promise。"""
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        run = self._active_run
        if run is not None and not run.promise.done():
            run.promise.set_result(None)
        self._active_run = None

    async def _process_events(self, event: AgentEvent) -> None:
        """先根据循环事件归约内部状态，再 await 各监听器。

        ``agent_end`` 只表示不会再有后续循环事件；要等该事件上被 await 的
        所有监听器结束、且 ``_finish_run()`` 清理运行时状态后，运行才算空闲。

        Args:
            event: 循环派发的 agent 事件。

        Raises:
            RuntimeError: 事件在活动运行之外被派发时。
        """
        if event.type == "message_start":
            self._state.streaming_message = event.message
        elif event.type == "message_update":
            self._state.streaming_message = event.message
        elif event.type == "message_end":
            self._state.streaming_message = None
            self._state.messages.append(event.message)
        elif event.type == "tool_execution_start":
            pending_tool_calls = set(self._state.pending_tool_calls)
            pending_tool_calls.add(event.tool_call_id)
            self._state.pending_tool_calls = pending_tool_calls
        elif event.type == "tool_execution_end":
            pending_tool_calls = set(self._state.pending_tool_calls)
            pending_tool_calls.discard(event.tool_call_id)
            self._state.pending_tool_calls = pending_tool_calls
        elif event.type == "turn_end":
            if getattr(event.message, "role", None) == "assistant" and getattr(event.message, "error_message", None):
                self._state.error_message = event.message.error_message
        elif event.type == "agent_end":
            self._state.streaming_message = None

        signal = self._active_run.abort_controller.signal if self._active_run is not None else None
        if signal is None:
            raise RuntimeError("Agent listener invoked outside active run")
        for listener in self._listeners:
            await maybe_await(listener(event, signal))
