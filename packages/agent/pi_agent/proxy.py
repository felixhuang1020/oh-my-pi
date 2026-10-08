"""供经服务器中转 LLM 调用的应用使用的代理 stream 函数。

移植自 ``packages/agent/src/proxy.ts``。SHTTP 接口基于 :mod:`httpx`：客户端由
:func:`pi_ai.utils.http.create_client` 构建，因此调用方提供的
:data:`~pi_ai.types.FetchFunction`（``fetch`` 选项）与其他已移植 provider 适配器
一样被尊重。TypeScript 对全局 ``fetch`` 打桩；本移植改为注入 fetch 可调用对象，
这正是代理测试可确定性离线运行的原因。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, fields
from typing import Any, TypeAlias

import httpx
from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Model,
    ProviderHeaders,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
    TranscriptContext,
    Usage,
)
from pi_ai.utils.event_stream import EventStream
from pi_ai.utils.events import (
    done_event,
    error_event,
    start_event,
    text_delta_event,
    text_end_event,
    text_start_event,
    thinking_delta_event,
    thinking_end_event,
    thinking_start_event,
    tool_call_delta_event,
    tool_call_end_event,
    tool_call_start_event,
)
from pi_ai.utils.http import create_client
from pi_ai.utils.json_parse import parse_streaming_json
from pi_ai.utils.serde import from_json, to_json

__all__ = [
    "ProxyAssistantMessageEvent",
    "ProxyDoneEvent",
    "ProxyErrorEvent",
    "ProxyMessageEventStream",
    "ProxySerializableStreamOptions",
    "ProxyStartEvent",
    "ProxyStreamOptions",
    "ProxyTextDeltaEvent",
    "ProxyTextEndEvent",
    "ProxyTextStartEvent",
    "ProxyThinkingDeltaEvent",
    "ProxyThinkingEndEvent",
    "ProxyThinkingStartEvent",
    "ProxyToolCallDeltaEvent",
    "ProxyToolCallEndEvent",
    "ProxyToolCallStartEvent",
    "build_proxy_request_options",
    "process_proxy_event",
    "stream_proxy",
]


def _now_ms() -> int:
    """返回当前时间戳（epoch 毫秒）。"""
    import time

    return int(time.time() * 1000)


def _empty_usage() -> Usage:
    """创建空的 usage 记录。"""
    return Usage()


def _is_terminal(event: Any) -> bool:
    """判断事件是否为终结事件。

    ``done`` 与 ``error`` 之外的事件都算非终结事件。

    Args:
        event: 待判断的代理事件。

    Returns:
        该事件是否为终结事件。
    """
    return event.type in ("done", "error")


def _extract_result(event: Any) -> AssistantMessage:
    """从终结事件中取出最终的 assistant 消息。

    Args:
        event: 终结事件，类型为 ``done`` 或 ``error``。

    Returns:
        成功时事件携带的 assistant 消息，或错误事件携带的 assistant 消息。

    Raises:
        ValueError: 事件类型不是 ``done`` 或 ``error`` 时。
    """
    if event.type == "done":
        return event.message
    if event.type == "error":
        return event.error
    raise ValueError("Unexpected event type")


class ProxyMessageEventStream(EventStream):
    """:func:`stream_proxy` 返回的事件流。"""

    def __init__(self) -> None:
        """初始化事件流，固定终结判定与结果提取策略。"""
        super().__init__(_is_terminal, _extract_result)


@dataclass
class ProxyStartEvent:
    """代理开始产出响应。

    Attributes:
        type: 事件类型标签，固定为 ``"start"``。
    """

    type: str = "start"


@dataclass
class ProxyTextStartEvent:
    """一个文本块开始。

    Attributes:
        content_index: 文本块在内容数组中的下标。
        type: 事件类型标签，固定为 ``"text_start"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    type: str = "text_start"


@dataclass
class ProxyTextDeltaEvent:
    """一个文本块增长。

    Attributes:
        content_index: 文本块在内容数组中的下标。
        delta: 本次新增的文本片段。
        type: 事件类型标签，固定为 ``"text_delta"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    delta: str = ""
    type: str = "text_delta"


@dataclass
class ProxyTextEndEvent:
    """一个文本块结束。

    Attributes:
        content_index: 文本块在内容数组中的下标。
        content_signature: 文本块签名；provider 未提供时为 ``None``。
        type: 事件类型标签，固定为 ``"text_end"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    content_signature: str | None = field(default=None, metadata={"alias": "contentSignature"})
    type: str = "text_end"


@dataclass
class ProxyThinkingStartEvent:
    """一个 thinking 块开始。

    Attributes:
        content_index: thinking 块在内容数组中的下标。
        type: 事件类型标签，固定为 ``"thinking_start"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    type: str = "thinking_start"


@dataclass
class ProxyThinkingDeltaEvent:
    """一个 thinking 块增长。

    Attributes:
        content_index: thinking 块在内容数组中的下标。
        delta: 本次新增的 thinking 片段。
        type: 事件类型标签，固定为 ``"thinking_delta"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    delta: str = ""
    type: str = "thinking_delta"


@dataclass
class ProxyThinkingEndEvent:
    """一个 thinking 块结束。

    Attributes:
        content_index: thinking 块在内容数组中的下标。
        content_signature: thinking 块签名；provider 未提供时为 ``None``。
        type: 事件类型标签，固定为 ``"thinking_end"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    content_signature: str | None = field(default=None, metadata={"alias": "contentSignature"})
    type: str = "thinking_end"


@dataclass
class ProxyToolCallStartEvent:
    """一个 tool call 开始。

    Attributes:
        content_index: tool call 在内容数组中的下标。
        id: 该 tool call 的 id。
        tool_name: 工具名。
        type: 事件类型标签，固定为 ``"toolcall_start"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    id: str = ""
    tool_name: str = field(default="", metadata={"alias": "toolName"})
    type: str = "toolcall_start"


@dataclass
class ProxyToolCallDeltaEvent:
    """tool call 的 JSON 参数增长。

    Attributes:
        content_index: tool call 在内容数组中的下标。
        delta: 本次新增的 JSON 参数片段。
        type: 事件类型标签，固定为 ``"toolcall_delta"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    delta: str = ""
    type: str = "toolcall_delta"


@dataclass
class ProxyToolCallEndEvent:
    """一个 tool call 结束。

    Attributes:
        content_index: tool call 在内容数组中的下标。
        tool_call: 参数已汇聚完成的 tool call 块。
        type: 事件类型标签，固定为 ``"toolcall_end"``。
    """

    content_index: int = field(default=0, metadata={"alias": "contentIndex"})
    tool_call: ToolCall = field(default_factory=ToolCall, metadata={"alias": "toolCall"})
    type: str = "toolcall_end"


@dataclass
class ProxyDoneEvent:
    """代理响应成功终止。

    Attributes:
        reason: 停止原因。
        usage: 本次调用的 token 用量。
        provider_thinking_level: provider 上报的 thinking 档位；未上报时为 ``None``。
        type: 事件类型标签，固定为 ``"done"``。
    """

    reason: StopReason | str = StopReason.STOP
    usage: Usage = field(default_factory=Usage)
    provider_thinking_level: str | None = field(default=None, metadata={"alias": "providerThinkingLevel"})
    type: str = "done"


@dataclass
class ProxyErrorEvent:
    """代理响应因错误或中止而终止。

    Attributes:
        reason: 停止原因。
        error_message: 面向用户的错误说明；无错误信息时为 ``None``。
        usage: 本次调用的 token 用量。
        provider_thinking_level: provider 上报的 thinking 档位；未上报时为 ``None``。
        type: 事件类型标签，固定为 ``"error"``。
    """

    reason: StopReason | str = StopReason.ERROR
    error_message: str | None = field(default=None, metadata={"alias": "errorMessage"})
    usage: Usage = field(default_factory=Usage)
    provider_thinking_level: str | None = field(default=None, metadata={"alias": "providerThinkingLevel"})
    type: str = "error"


#: 服务器发送的代理事件，delta 类事件会剥除 partial 字段以节省带宽。
ProxyAssistantMessageEvent: TypeAlias = (
    ProxyStartEvent
    | ProxyTextStartEvent
    | ProxyTextDeltaEvent
    | ProxyTextEndEvent
    | ProxyThinkingStartEvent
    | ProxyThinkingDeltaEvent
    | ProxyThinkingEndEvent
    | ProxyToolCallStartEvent
    | ProxyToolCallDeltaEvent
    | ProxyToolCallEndEvent
    | ProxyDoneEvent
    | ProxyErrorEvent
)


@dataclass
class ProxySerializableStreamOptions:
    """发送给代理服务器的 :class:`~pi_ai.types.SimpleStreamOptions` 子集。

    Attributes:
        temperature: 采样温度。
        sampling_params: provider 特有的采样参数。
        max_tokens: 单次响应的最大 token 数。
        reasoning: reasoning/thinking 相关的请求配置。
        cache_retention: prompt 缓存保留策略。
        session_id: 会话 id，供服务端做缓存或关联。
        headers: 附加到请求上的 provider 头。
        metadata: 请求附带的元数据。
        transport: provider 特有的传输配置。
        thinking_budgets: 各 thinking 档位对应的 token 预算。
        max_retry_delay_ms: 两次重试之间的最大等待毫秒数。
    """

    temperature: float | None = None
    sampling_params: dict[str, Any] | None = field(default=None, metadata={"alias": "samplingParams"})
    max_tokens: int | None = field(default=None, metadata={"alias": "maxTokens"})
    reasoning: Any = None
    cache_retention: Any = field(default=None, metadata={"alias": "cacheRetention"})
    session_id: str | None = field(default=None, metadata={"alias": "sessionId"})
    headers: ProviderHeaders | None = None
    metadata: dict[str, Any] | None = None
    transport: Any = None
    thinking_budgets: Any = field(default=None, metadata={"alias": "thinkingBudgets"})
    max_retry_delay_ms: int | None = field(default=None, metadata={"alias": "maxRetryDelayMs"})


@dataclass
class ProxyStreamOptions(ProxySerializableStreamOptions):
    """:func:`stream_proxy` 的选项。

    在可序列化子集之外，另带仅本地使用的连接与鉴权字段。

    Attributes:
        signal: 代理请求的本地 abort signal。
        auth_token: 代理服务器的鉴权 token。
        proxy_url: 代理服务器 URL（如 ``"https://genai.example.com"``）。
        fetch: 可选的传输注入，对应 TypeScript 对全局 ``fetch`` 的打桩。
        timeout_ms: 可选的单次请求超时（毫秒）。
    """

    #: 代理请求的本地 abort signal。
    signal: Any = None
    #: 代理服务器的鉴权 token。
    auth_token: str = field(default="", metadata={"alias": "authToken"})
    #: 代理服务器 URL（如 ``"https://genai.example.com"``）。
    proxy_url: str = field(default="", metadata={"alias": "proxyUrl"})
    #: 可选的传输注入，对应 TypeScript 对全局 ``fetch`` 的打桩。
    fetch: Any = None
    #: 可选的单次请求超时（毫秒）。
    timeout_ms: int | None = field(default=None, metadata={"alias": "timeoutMs"})


def build_proxy_request_options(options: ProxyStreamOptions) -> ProxySerializableStreamOptions:
    """把完整代理选项投影为可序列化、发送给服务器的子集。

    Args:
        options: 调用方传入的完整代理选项。

    Returns:
        仅含可序列化字段、可直接 JSON 化的选项对象。
    """
    return ProxySerializableStreamOptions(
        temperature=options.temperature,
        sampling_params=options.sampling_params,
        max_tokens=options.max_tokens,
        reasoning=options.reasoning,
        cache_retention=options.cache_retention,
        session_id=options.session_id,
        headers=options.headers,
        metadata=options.metadata,
        transport=options.transport,
        thinking_budgets=options.thinking_budgets,
        max_retry_delay_ms=options.max_retry_delay_ms,
    )


def stream_proxy(
    model: Model,
    context: TranscriptContext,
    options: ProxyStreamOptions,
) -> ProxyMessageEventStream:
    """经由服务器中转、而非直连 LLM provider 的 stream 函数。

    服务器会从 delta 事件中剥除 ``partial`` 字段以节省带宽；
    partial 消息在客户端重建。

    创建需要走代理的 :class:`~pi_agent.agent.Agent` 时，将本函数用作
    ``stream_fn`` 选项。

    Args:
        model: 要调用的模型。
        context: 规范化后的会话记录。
        options: 代理连接、鉴权与传输选项。

    Returns:
        承载本次响应事件的 :class:`ProxyMessageEventStream`。
    """
    stream = ProxyMessageEventStream()
    task = asyncio.ensure_future(_drive_proxy(stream, model, context, options))
    task.add_done_callback(_swallow)
    return stream


def _swallow(task: asyncio.Future[Any]) -> None:
    """消耗后台任务的异常，避免其被判定为「异常从未取出」。

    Args:
        task: 已结束的后台任务；任务被取消时直接返回。
    """
    if task.cancelled():
        return
    task.exception()


async def _safe_aclose(response: httpx.Response) -> None:
    """关闭响应，关闭过程中抛出的异常一律忽略。

    Args:
        response: 要关闭的 httpx 响应。
    """
    try:
        await response.aclose()
    except Exception:  # noqa: BLE001 - 关闭已中止的响应绝不能抛出
        pass


async def _drive_proxy(
    stream: ProxyMessageEventStream,
    model: Model,
    context: TranscriptContext,
    options: ProxyStreamOptions,
) -> None:
    """执行一次代理请求，把 SSE 帧翻译为 assistant 消息事件。

    该协程在后台任务中运行，所有故障都会编码进 ``stream`` 而不向外抛出。

    Args:
        stream: 事件流，用于把翻译后的事件推送给消费者。
        model: 要调用的模型。
        context: 规范化后的会话记录。
        options: 代理连接、鉴权与传输选项。
    """
    # 初始化 partial 消息，后续由事件逐步构建
    partial = AssistantMessage(
        role="assistant",
        stop_reason=StopReason.PENDING,
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=_empty_usage(),
        timestamp=_now_ms(),
    )

    response_holder: dict[str, httpx.Response] = {}

    def abort_handler() -> None:
        """abort signal 触发时，把已建立的响应异步关闭。"""
        response = response_holder.get("response")
        if response is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - 中止均发生在运行中的事件循环里
            return
        loop.create_task(_safe_aclose(response))

    signal = options.signal
    if signal is not None:
        signal.add_listener(abort_handler)

    try:
        headers = {
            "Authorization": f"Bearer {options.auth_token}",
            "Content-Type": "application/json",
        }
        body = json.dumps(
            to_json(
                {
                    "model": model,
                    "context": context,
                    "options": build_proxy_request_options(options),
                }
            )
        ).encode("utf-8")

        async with create_client(headers=headers, fetch=options.fetch, timeout_ms=options.timeout_ms) as client:
            async with client.stream("POST", f"{options.proxy_url}/api/stream", content=body) as response:
                response_holder["response"] = response

                if not response.is_success:
                    error_message = f"Proxy error: {response.status_code} {response.reason_phrase}"
                    try:
                        raw = (await response.aread()).decode(response.encoding or "utf-8", errors="replace")
                        error_data = json.loads(raw)
                        if isinstance(error_data, dict) and error_data.get("error"):
                            error_message = f"Proxy error: {error_data['error']}"
                    except Exception:  # noqa: BLE001 - 无法解析错误响应体
                        pass
                    raise RuntimeError(error_message)

                saw_terminal_event = False
                async for line in response.aiter_lines():
                    if signal is not None and signal.aborted:
                        raise RuntimeError("Request aborted by user")
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if not data:
                        continue
                    proxy_event = from_json(ProxyAssistantMessageEvent, json.loads(data))
                    event = process_proxy_event(proxy_event, partial)
                    if event is not None:
                        if event.type in ("done", "error"):
                            saw_terminal_event = True
                        stream.push(event)

                if signal is not None and signal.aborted:
                    raise RuntimeError("Request aborted by user")

                if not saw_terminal_event:
                    # 没有 done/error 事件的正常 EOF 意味着服务器中途断开了响应。
                    # 应作为错误呈现，而不是让消费者一直等一个永不到来的结果。
                    partial.stop_reason = "error"
                    partial.error_message = "Connection closed by proxy server before the response completed"
                    stream.push(error_event("error", partial))

                stream.end()
    except Exception as error:  # noqa: BLE001 - 故障统一编码进 stream
        error_message = str(error)
        aborted = signal is not None and signal.aborted
        reason = StopReason.ABORTED if aborted else StopReason.ERROR
        partial.stop_reason = reason
        partial.error_message = error_message
        stream.push(error_event(reason, partial))
        stream.end()
    finally:
        if signal is not None:
            signal.remove_listener(abort_handler)


def process_proxy_event(
    proxy_event: ProxyAssistantMessageEvent,
    partial: AssistantMessage,
) -> AssistantMessageEvent | None:
    """处理一个代理事件并更新 partial 消息。

    Args:
        proxy_event: 服务器下发的单个代理事件。
        partial: 正在重建的 partial assistant 消息，会被原地修改。

    Returns:
        对应的 assistant 消息事件；该事件无需额外产出时返回 ``None``。

    Raises:
        ValueError: delta/end 事件命中的内容块类型与事件不匹配时。
    """
    event_type = proxy_event.type

    if event_type == "start":
        return start_event(partial)

    if event_type == "text_start":
        _set_content(partial, proxy_event.content_index, TextContent(text=""))
        return text_start_event(proxy_event.content_index, partial)

    if event_type == "text_delta":
        content = _content_at(partial, proxy_event.content_index)
        if isinstance(content, TextContent):
            content.text += proxy_event.delta
            return text_delta_event(proxy_event.content_index, proxy_event.delta, partial)
        raise ValueError("Received text_delta for non-text content")

    if event_type == "text_end":
        content = _content_at(partial, proxy_event.content_index)
        if isinstance(content, TextContent):
            content.text_signature = proxy_event.content_signature
            return text_end_event(proxy_event.content_index, content.text, partial)
        raise ValueError("Received text_end for non-text content")

    if event_type == "thinking_start":
        _set_content(partial, proxy_event.content_index, ThinkingContent(thinking=""))
        return thinking_start_event(proxy_event.content_index, partial)

    if event_type == "thinking_delta":
        content = _content_at(partial, proxy_event.content_index)
        if isinstance(content, ThinkingContent):
            content.thinking += proxy_event.delta
            return thinking_delta_event(proxy_event.content_index, proxy_event.delta, partial)
        raise ValueError("Received thinking_delta for non-thinking content")

    if event_type == "thinking_end":
        content = _content_at(partial, proxy_event.content_index)
        if isinstance(content, ThinkingContent):
            content.thinking_signature = proxy_event.content_signature
            return thinking_end_event(proxy_event.content_index, content.thinking, partial)
        raise ValueError("Received thinking_end for non-thinking content")

    if event_type == "toolcall_start":
        tool_call = ToolCall(
            type="toolCall",
            id=proxy_event.id,
            name=proxy_event.tool_name,
            arguments={},
        )
        _set_content(partial, proxy_event.content_index, tool_call)
        setattr(tool_call, "partial_json", "")
        return tool_call_start_event(proxy_event.content_index, partial)

    if event_type == "toolcall_delta":
        content = _content_at(partial, proxy_event.content_index)
        if isinstance(content, ToolCall):
            partial_json = getattr(content, "partial_json", "") + proxy_event.delta
            setattr(content, "partial_json", partial_json)
            content.arguments = parse_streaming_json(partial_json) or {}
            _set_content(partial, proxy_event.content_index, content)  # 触发响应式更新
            return tool_call_delta_event(proxy_event.content_index, proxy_event.delta, partial)
        raise ValueError("Received toolcall_delta for non-toolCall content")

    if event_type == "toolcall_end":
        content = _content_at(partial, proxy_event.content_index)
        if isinstance(content, ToolCall):
            for dataclass_field in fields(ToolCall):
                setattr(content, dataclass_field.name, getattr(proxy_event.tool_call, dataclass_field.name))
            if hasattr(content, "partial_json"):
                delattr(content, "partial_json")
            return tool_call_end_event(proxy_event.content_index, content, partial)
        return None

    if event_type == "done":
        partial.stop_reason = proxy_event.reason
        partial.usage = proxy_event.usage
        if proxy_event.provider_thinking_level is not None:
            partial.provider_thinking_level = proxy_event.provider_thinking_level
        return done_event(proxy_event.reason, partial)

    if event_type == "error":
        partial.stop_reason = proxy_event.reason
        partial.error_message = proxy_event.error_message
        partial.usage = proxy_event.usage
        if proxy_event.provider_thinking_level is not None:
            partial.provider_thinking_level = proxy_event.provider_thinking_level
        return error_event(proxy_event.reason, partial)

    print(f"Unhandled proxy event type: {event_type}")
    return None


def _content_at(partial: AssistantMessage, index: int) -> Any:
    """按下标读取 partial 消息的内容块。

    Args:
        partial: 正在重建的 partial assistant 消息。
        index: 内容块下标。

    Returns:
        对应的内容块；下标越界时返回 ``None``。
    """
    if index < 0 or index >= len(partial.content):
        return None
    return partial.content[index]


def _set_content(partial: AssistantMessage, index: int, block: Any) -> None:
    """赋值 ``partial.content[index]``，按 JavaScript 数组语义自动扩容。

    Args:
        partial: 正在重建的 partial assistant 消息。
        index: 要写入的内容块下标。
        block: 要写入的内容块。
    """
    while len(partial.content) <= index:
        partial.content.append(None)
    partial.content[index] = block
