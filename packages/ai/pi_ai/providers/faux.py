"""测试用的 faux provider。

移植自 ``packages/ai/src/providers/faux.ts``。这是一个脚本化 provider，响应预先
排队；它 stream 确定性的 delta，让测试无需联网即可覆盖整个
``AssistantMessageEventStream`` 协议。
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias

from ..auth.types import AuthResult, ModelAuth, ProviderAuth
from ..models import CreateProviderOptions, ProviderImpl, create_provider
from ..types import (
    AssistantMessage,
    AssistantMessageEventStream,
    Cost,
    DeferredCancelOptions,
    DeferredFetchOptions,
    DeferredHandle,
    JsonObject,
    Message,
    Model,
    ModelCost,
    ModelInputLimits,
    ProviderResponse,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Usage,
)
from ..utils.async_utils import maybe_await
from ..utils.event_stream import create_assistant_message_event_stream
from ..utils.events import (
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
from ..utils.serde import clone, from_json, to_json
from ..utils.text import get_system_message_text

__all__ = [
    "FauxContentBlock",
    "FauxCore",
    "FauxModelDefinition",
    "FauxProviderHandle",
    "FauxProviderRegistration",
    "FauxProviderState",
    "FauxResponseFactory",
    "FauxResponseStep",
    "RegisterFauxProviderOptions",
    "create_faux_core",
    "faux_assistant_message",
    "faux_provider",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
]

DEFAULT_API = "faux"
DEFAULT_PROVIDER = "faux"
DEFAULT_MODEL_ID = "faux-1"
DEFAULT_MODEL_NAME = "Faux Model"
DEFAULT_BASE_URL = "http://localhost:0"
DEFAULT_MIN_TOKEN_SIZE = 3
DEFAULT_MAX_TOKEN_SIZE = 5

DEFAULT_USAGE = Usage(
    input=0,
    output=0,
    cache_read=0,
    cache_write=0,
    total_tokens=0,
    cost=Cost(),
)

_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"


# --------------------------------------------------------------------------------------
# 内容构造函数
# --------------------------------------------------------------------------------------


@dataclass
class FauxModelDefinition:
    """faux provider 对外暴露的其中一个模型。

    Attributes:
        id: 模型 id。
        name: 展示名称；为 ``None`` 时回退到 ``id``。
        reasoning: 是否支持推理（thinking）。
        input: 支持的输入类型列表，例如 ``["text"]``。
        input_limits: 输入长度限制。
        cost: 价格配置，可为 :class:`ModelCost` 或原始字典。
        context_window: 上下文窗口大小（token）。
        max_tokens: 单次响应的最大输出 token 数。
    """

    id: str = ""
    name: str | None = None
    reasoning: bool | None = None
    input: list[str] | None = None
    input_limits: ModelInputLimits | None = field(default=None, metadata={"alias": "inputLimits"})
    cost: ModelCost | dict[str, float] | None = None
    context_window: int | None = field(default=None, metadata={"alias": "contextWindow"})
    max_tokens: int | None = field(default=None, metadata={"alias": "maxTokens"})


#: faux 响应可能包含的内容块。
FauxContentBlock: TypeAlias = TextContent | ThinkingContent | ToolCall


def faux_text(text: str) -> TextContent:
    """纯文本块。

    Args:
        text: 文本内容。

    Returns:
        包装该文本的 :class:`TextContent`。
    """
    return TextContent(type="text", text=text)


def faux_thinking(thinking: str) -> ThinkingContent:
    """推理（thinking）块。

    Args:
        thinking: 推理文本。

    Returns:
        包装该推理文本的 :class:`ThinkingContent`。
    """
    return ThinkingContent(type="thinking", thinking=thinking)


def faux_tool_call(name: str, arguments: JsonObject, options: dict[str, Any] | None = None) -> ToolCall:
    """工具调用块，可由调用方指定 id。

    Args:
        name: 工具名称。
        arguments: 工具参数。
        options: 可选参数，``id`` 键用于指定调用 id；为 ``None`` 或未提供
            ``id`` 时随机生成。

    Returns:
        构造出的 :class:`ToolCall`。
    """
    options = options or {}
    return ToolCall(
        type="toolCall",
        id=options.get("id") or _random_id("tool"),
        name=name,
        arguments=arguments,
    )


def normalize_faux_assistant_content(content: str | FauxContentBlock | list[FauxContentBlock]) -> list[FauxContentBlock]:
    """把字符串/单块/列表统一规约为块列表。

    Args:
        content: 字符串、单个内容块或内容块列表。

    Returns:
        内容块列表（字符串会被包装为文本块）。
    """
    if isinstance(content, str):
        return [faux_text(content)]
    return list(content) if isinstance(content, list) else [content]


def faux_assistant_message(
    content: str | FauxContentBlock | list[FauxContentBlock],
    *,
    stop_reason: StopReason | str | None = None,
    deferred: DeferredHandle | None = None,
    error_message: str | None = None,
    response_id: str | None = None,
    timestamp: int | None = None,
) -> AssistantMessage:
    """携带 faux 默认值的脚本化 assistant 消息。

    Args:
        content: 消息内容，允许字符串、单个内容块或内容块列表。
        stop_reason: 停止原因；为 ``None`` 时使用 ``"stop"``。
        deferred: 延迟响应句柄；为 ``None`` 表示非延迟响应。
        error_message: 错误说明；仅在错误/中止终止态下使用。
        response_id: 上游响应 id。
        timestamp: 创建时间（epoch 毫秒）；为 ``None`` 时使用当前时间。

    Returns:
        使用 faux 默认 api、provider 与模型 id 填充的 assistant 消息。
    """
    return AssistantMessage(
        role="assistant",
        content=normalize_faux_assistant_content(content),
        api=DEFAULT_API,
        provider=DEFAULT_PROVIDER,
        model=DEFAULT_MODEL_ID,
        usage=DEFAULT_USAGE,
        stop_reason=stop_reason if stop_reason is not None else "stop",
        deferred=deferred,
        error_message=error_message,
        response_id=response_id,
        timestamp=timestamp if timestamp is not None else _now_ms(),
    )


# --------------------------------------------------------------------------------------
# 测试脚手架类型
# --------------------------------------------------------------------------------------


@dataclass
class FauxProviderState:
    """测试可断言的可变计数器。

    Attributes:
        call_count: :meth:`FauxCore.stream` 的累计调用次数。
        deferred_fetch_count: :meth:`FauxCore.fetch_deferred` 的累计调用次数。
        cancelled_deferred: 已取消的延迟响应句柄快照列表。
    """

    call_count: int = 0
    deferred_fetch_count: int = 0
    cancelled_deferred: list[DeferredHandle] = field(default_factory=list)


#: 根据请求上下文生成下一条脚本化响应。
FauxResponseFactory: TypeAlias = Callable[
    [TranscriptContext, SimpleStreamOptions | None, FauxProviderState, Model],
    "AssistantMessage | Awaitable[AssistantMessage]",
]
#: 入队的一条响应：消息或工厂函数。
FauxResponseStep: TypeAlias = AssistantMessage | FauxResponseFactory


@dataclass
class FauxDeferredOptions:
    """延迟（deferred）响应的脚本化开关。"""

    #: 脚本化响应就绪前，返回原始 handle 的 fetch 次数。
    pending_fetches: int | None = field(default=None, metadata={"alias": "pendingFetches"})
    poll_after_ms: int | None = field(default=None, metadata={"alias": "pollAfterMs"})


@dataclass
class FauxTokenSize:
    """delta 分片大小的上下限，单位为 token。

    Attributes:
        min: 每个分片的最小 token 数；为 ``None`` 时使用默认值。
        max: 每个分片的最大 token 数；为 ``None`` 时使用默认值。
    """

    min: int | None = None
    max: int | None = None


@dataclass
class RegisterFauxProviderOptions:
    """:func:`create_faux_core` / :func:`faux_provider` 的选项。

    Attributes:
        api: 覆盖 API 标识；为 ``None`` 时随机生成。
        provider: 覆盖 provider id；为 ``None`` 时使用默认 faux provider。
        models: 模型定义列表；为 ``None`` 时使用单个默认模型。
        deferred: 延迟响应的脚本化开关。
        tokens_per_second: 模拟的 stream 速率；为 ``None`` 时不限速。
        token_size: delta 分片大小范围。
    """

    api: str | None = None
    provider: str | None = None
    models: list[FauxModelDefinition] | None = None
    deferred: FauxDeferredOptions | None = None
    tokens_per_second: int | None = field(default=None, metadata={"alias": "tokensPerSecond"})
    token_size: FauxTokenSize | None = field(default=None, metadata={"alias": "tokenSize"})


@dataclass
class FauxProviderHandle:
    """faux provider 及其脚本化接口，供 ``createModels()`` 使用。

    Attributes:
        provider: 创建好的 :class:`ProviderImpl`。
        api: provider 的 API 标识。
        models: 已解析的模型列表。
        get_model: 按 id 取模型的函数。
        state: 可断言的调用计数器。
        set_responses: 替换脚本化响应队列的函数。
        append_responses: 追加脚本化响应的函数。
        get_pending_response_count: 查询队列剩余条数的函数。
    """

    provider: ProviderImpl
    api: str
    models: list[Model]
    get_model: Callable[..., Model | None]
    state: FauxProviderState
    set_responses: Callable[[list[FauxResponseStep]], None]
    append_responses: Callable[[list[FauxResponseStep]], None]
    get_pending_response_count: Callable[[], int]


@dataclass
class FauxProviderRegistration:
    """注册进 compat api 注册表的 faux provider。

    Attributes:
        api: provider 的 API 标识。
        models: 已解析的模型列表。
        get_model: 按 id 取模型的函数。
        state: 可断言的调用计数器。
        set_responses: 替换脚本化响应队列的函数。
        append_responses: 追加脚本化响应的函数。
        get_pending_response_count: 查询队列剩余条数的函数。
        unregister: 从注册表移除该 provider 的函数。
    """

    api: str
    models: list[Model]
    get_model: Callable[..., Model | None]
    state: FauxProviderState
    set_responses: Callable[[list[FauxResponseStep]], None]
    append_responses: Callable[[list[FauxResponseStep]], None]
    get_pending_response_count: Callable[[], int]
    unregister: Callable[[], None]


# --------------------------------------------------------------------------------------
# 内部实现
# --------------------------------------------------------------------------------------


def _now_ms() -> int:
    """当前时间的 epoch 毫秒表示。"""
    return int(time.time() * 1000)


def _random_suffix() -> str:
    """生成 52 位随机数的 base36 表示。

    Returns:
        不含前导零的 base36 字符串；随机数为 0 时返回 ``"0"``。
    """
    value = random.getrandbits(52)
    if value == 0:
        return "0"
    digits: list[str] = []
    while value:
        value, remainder = divmod(value, 36)
        digits.append(_BASE36[remainder])
    return "".join(reversed(digits))


def _random_id(prefix: str) -> str:
    """生成带前缀的随机 id。

    Args:
        prefix: id 前缀，例如 ``"tool"``。

    Returns:
        形如 ``prefix:时间戳:随机后缀`` 的字符串。
    """
    return f"{prefix}:{_now_ms()}:{_random_suffix()}"


def _json_compact(value: Any) -> str:
    """把值序列化为紧凑 JSON。

    Args:
        value: 任意可序列化的值。

    Returns:
        使用最短分隔符、保留非 ASCII 字符的 JSON 字符串。
    """
    return json.dumps(to_json(value), separators=(",", ":"), ensure_ascii=False)


def estimate_tokens(text: str) -> int:
    """faux provider 的 token 估算：每四个字符算一个 token。

    Args:
        text: 待估算文本。

    Returns:
        向上取整后的 token 数。
    """
    return math.ceil(len(text) / 4)


def content_to_text(content: str | list[TextContent]) -> str:
    """把 user 内容渲染为文本。

    Args:
        content: 字符串或文本内容块列表。

    Returns:
        拼接后的文本。
    """
    if isinstance(content, str):
        return content
    return "\n".join(block.text for block in content)


def assistant_content_to_text(content: list[TextContent | ThinkingContent | ToolCall]) -> str:
    """把 assistant 内容渲染为文本，工具参数序列化为紧凑 JSON。

    Args:
        content: assistant 内容块列表。

    Returns:
        拼接后的文本；工具调用渲染为 ``名称:参数``。
    """
    parts = []
    for block in content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "thinking":
            parts.append(block.thinking)
        else:
            parts.append(f"{block.name}:{_json_compact(block.arguments)}")
    return "\n".join(parts)


def tool_result_to_text(message: ToolResultMessage) -> str:
    """把工具结果消息渲染为文本。

    Args:
        message: 工具结果消息。

    Returns:
        工具名称与其各内容块文本拼接后的结果。
    """
    return "\n".join([message.tool_name, *[content_to_text([block]) for block in message.content]])


def message_to_text(message: Message) -> str:
    """为 prompt-cache 估算渲染单条 transcript 消息。

    Args:
        message: 任意角色的 transcript 消息。

    Returns:
        该消息的文本表示；system 消息还会带上工具的增删记录。
    """
    if message.role == "system":
        parts = [
            get_system_message_text(message),
            *[f"tool-:{_json_compact(tool)}" for tool in (message.tools_removed or [])],
            *[f"tool+:{_json_compact(tool)}" for tool in (message.tools_added or [])],
        ]
        return "\n".join(part for part in parts if len(part) > 0)
    if message.role == "user":
        return content_to_text(message.content)
    if message.role == "assistant":
        return assistant_content_to_text(message.content)
    return tool_result_to_text(message)


def serialize_context(context: TranscriptContext) -> str:
    """序列化 transcript，用于 prompt-cache 前缀比较。

    Args:
        context: 本次请求的 transcript 上下文。

    Returns:
        每条消息按 ``角色:文本`` 拼接、以空行分隔的字符串。
    """
    return "\n\n".join(f"{message.role}:{message_to_text(message)}" for message in context.messages)


def common_prefix_length(a: str, b: str) -> int:
    """最长公共前缀的长度。

    Args:
        a: 第一个字符串。
        b: 第二个字符串。

    Returns:
        两字符串从头开始相同的字符数。
    """
    length = min(len(a), len(b))
    index = 0
    while index < length and a[index] == b[index]:
        index += 1
    return index


def with_usage_estimate(
    message: AssistantMessage,
    context: TranscriptContext,
    options: StreamOptions | None,
    prompt_cache: dict[str, str],
) -> AssistantMessage:
    """填充 usage，含模拟的 prompt-cache 读/写拆分。

    以 session id 为键复用上一次的 prompt 文本，用公共前缀长度模拟
    缓存命中，未命中的部分记为 cache write。

    Args:
        message: 待填充 usage 的 assistant 消息。
        context: 本次请求的 transcript 上下文。
        options: stream 选项，用于读取 session id 与 cache retention。
        prompt_cache: 以 session id 为键的上一次 prompt 文本缓存，会被就地更新。

    Returns:
        带估算 usage 的新 assistant 消息。
    """
    prompt_text = serialize_context(context)
    prompt_tokens = estimate_tokens(prompt_text)
    output_tokens = estimate_tokens(assistant_content_to_text(message.content))
    input_tokens = prompt_tokens
    cache_read = 0
    cache_write = 0
    session_id = options.session_id if options is not None else None

    if session_id and (options is None or options.cache_retention != "none"):
        previous_prompt = prompt_cache.get(session_id)
        if previous_prompt:
            cached_chars = common_prefix_length(previous_prompt, prompt_text)
            cache_read = estimate_tokens(previous_prompt[:cached_chars])
            cache_write = estimate_tokens(prompt_text[cached_chars:])
            input_tokens = max(0, prompt_tokens - cache_read)
        else:
            cache_write = prompt_tokens
        prompt_cache[session_id] = prompt_text

    return replace(
        message,
        usage=Usage(
            input=input_tokens,
            output=output_tokens,
            cache_read=cache_read,
            cache_write=cache_write,
            total_tokens=input_tokens + output_tokens + cache_read + cache_write,
            cost=Cost(),
        ),
    )


def split_string_by_token_size(text: str, min_token_size: int, max_token_size: int) -> list[str]:
    """把文本切成随机大小、近似 token 边界的分片。

    Args:
        text: 待切分文本。
        min_token_size: 每个分片的最小 token 数。
        max_token_size: 每个分片的最大 token 数。

    Returns:
        分片列表；空文本返回包含一个空字符串的列表。
    """
    chunks: list[str] = []
    index = 0
    while index < len(text):
        token_size = min_token_size + math.floor(random.random() * (max_token_size - min_token_size + 1))
        char_size = max(1, token_size * 4)
        chunks.append(text[index : index + char_size])
        index += char_size
    return chunks if chunks else [""]


def _copy(message: AssistantMessage) -> AssistantMessage:
    """浅拷贝消息，避免事件流中的增量修改影响原消息。

    Args:
        message: 待拷贝的消息。

    Returns:
        字段相同的新消息对象。
    """
    return replace(message)


def clone_message(message: AssistantMessage, api: str, provider: str, model_id: str) -> AssistantMessage:
    """深克隆脚本化消息，并重新绑定到发起请求的模型。

    Args:
        message: 脚本化消息。
        api: 目标 API 标识。
        provider: 目标 provider id。
        model_id: 目标模型 id。

    Returns:
        绑定到目标模型、且时间戳与 usage 已补齐的新消息。
    """
    cloned = clone(message)
    return replace(
        cloned,
        api=api,
        provider=provider,
        model=model_id,
        timestamp=cloned.timestamp or _now_ms(),
        usage=cloned.usage or DEFAULT_USAGE,
    )


def create_deferred_message(model: Model, handle: DeferredHandle) -> AssistantMessage:
    """延迟响应的第一轮消息。

    Args:
        model: 发起请求的模型。
        handle: 本次延迟响应的句柄。

    Returns:
        空内容、``stop_reason`` 为 ``"deferred"`` 的 assistant 消息。
    """
    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=DEFAULT_USAGE,
        stop_reason="deferred",
        deferred=handle,
        timestamp=_now_ms(),
    )


def create_error_message(error: Any, api: str, provider: str, model_id: str) -> AssistantMessage:
    """把失败编码为终止态 assistant 消息。

    Args:
        error: 原始异常或错误值，会被转为字符串写入 ``error_message``。
        api: 目标 API 标识。
        provider: 目标 provider id。
        model_id: 目标模型 id。

    Returns:
        ``stop_reason`` 为 ``"error"`` 的 assistant 消息。
    """
    return AssistantMessage(
        role="assistant",
        content=[],
        api=api,
        provider=provider,
        model=model_id,
        usage=DEFAULT_USAGE,
        stop_reason="error",
        error_message=str(error),
        timestamp=_now_ms(),
    )


def create_aborted_message(partial: AssistantMessage) -> AssistantMessage:
    """把部分消息转换为中止态的终止消息。

    Args:
        partial: 已推送部分内容的 assistant 消息。

    Returns:
        ``stop_reason`` 为 ``"aborted"``、带中止说明的新消息。
    """
    return replace(partial, stop_reason="aborted", error_message="Request was aborted", timestamp=_now_ms())


async def schedule_chunk(chunk: str, tokens_per_second: int | None) -> None:
    """按配置的 stream 速率，等待一个分片应有的时长。

    Args:
        chunk: 即将推送的分片文本。
        tokens_per_second: 模拟速率；为 ``None`` 或非正数时只让出事件循环。
    """
    if not tokens_per_second or tokens_per_second <= 0:
        await asyncio.sleep(0)
        return
    delay_ms = (estimate_tokens(chunk) / tokens_per_second) * 1000
    await asyncio.sleep(delay_ms / 1000)


async def stream_with_deltas(
    stream: AssistantMessageEventStream,
    message: AssistantMessage,
    min_token_size: int,
    max_token_size: int,
    tokens_per_second: int | None,
    signal: Any = None,
) -> None:
    """推送 ``message`` 完整的 start/delta/end 事件序列。

    逐个内容块把文本或工具参数切成随机分片，按 ``tokens_per_second`` 节流后
    以 delta 事件推送，每推一片都会检查取消信号。终止态由
    ``message.stop_reason`` 决定：``"error"`` 与 ``"aborted"`` 推送 error
    事件，其余推送 done。

    Args:
        stream: 目标 assistant 消息事件 stream。
        message: 已解析完成的脚本化消息。
        min_token_size: delta 分片的最小 token 数。
        max_token_size: delta 分片的最大 token 数。
        tokens_per_second: 模拟的 stream 速率；为 ``None`` 时不限速。
        signal: 取消信号；为 ``None`` 时不检查取消。

    Raises:
        RuntimeError: ``message.stop_reason`` 仍为 ``"pending"``（脚本化响应
            缺少停止原因）时。
    """
    partial = replace(message, content=[], stop_reason="pending")
    if signal is not None and signal.aborted:
        aborted = create_aborted_message(partial)
        stream.push(error_event("aborted", aborted))
        stream.end(aborted)
        return

    stream.push(start_event(_copy(partial)))

    for index, block in enumerate(message.content):
        if signal is not None and signal.aborted:
            aborted = create_aborted_message(partial)
            stream.push(error_event("aborted", aborted))
            stream.end(aborted)
            return

        if block.type == "thinking":
            partial.content = [*partial.content, ThinkingContent(type="thinking", thinking="")]
            stream.push(thinking_start_event(index, _copy(partial)))
            for chunk in split_string_by_token_size(block.thinking, min_token_size, max_token_size):
                await schedule_chunk(chunk, tokens_per_second)
                if signal is not None and signal.aborted:
                    aborted = create_aborted_message(partial)
                    stream.push(error_event("aborted", aborted))
                    stream.end(aborted)
                    return
                partial.content[index].thinking += chunk  # type: ignore[union-attr]
                stream.push(thinking_delta_event(index, chunk, _copy(partial)))
            stream.push(thinking_end_event(index, block.thinking, _copy(partial)))
            continue

        if block.type == "text":
            partial.content = [*partial.content, TextContent(type="text", text="")]
            stream.push(text_start_event(index, _copy(partial)))
            for chunk in split_string_by_token_size(block.text, min_token_size, max_token_size):
                await schedule_chunk(chunk, tokens_per_second)
                if signal is not None and signal.aborted:
                    aborted = create_aborted_message(partial)
                    stream.push(error_event("aborted", aborted))
                    stream.end(aborted)
                    return
                partial.content[index].text += chunk  # type: ignore[union-attr]
                stream.push(text_delta_event(index, chunk, _copy(partial)))
            stream.push(text_end_event(index, block.text, _copy(partial)))
            continue

        partial.content = [
            *partial.content,
            ToolCall(type="toolCall", id=block.id, name=block.name, arguments={}),
        ]
        stream.push(tool_call_start_event(index, _copy(partial)))
        for chunk in split_string_by_token_size(_json_compact(block.arguments), min_token_size, max_token_size):
            await schedule_chunk(chunk, tokens_per_second)
            if signal is not None and signal.aborted:
                aborted = create_aborted_message(partial)
                stream.push(error_event("aborted", aborted))
                stream.end(aborted)
                return
            stream.push(tool_call_delta_event(index, chunk, _copy(partial)))
        partial.content[index].arguments = block.arguments  # type: ignore[union-attr]
        stream.push(tool_call_end_event(index, block, _copy(partial)))

    if message.stop_reason == "pending":
        raise RuntimeError("Faux response ended without a stop reason")
    if message.stop_reason in ("error", "aborted"):
        stream.push(error_event(message.stop_reason, message))
        stream.end(message)
        return

    stream.push(done_event(message.stop_reason, message))
    stream.end(message)


def _swallow(task: asyncio.Task) -> None:
    """消费后台任务的异常，避免「exception never retrieved」告警。

    Args:
        task: 已结束的后台任务。
    """
    if task.cancelled():
        return
    task.exception()


@dataclass
class _DeferredEntry:
    """尚未提交的延迟响应在内存中的状态。

    Attributes:
        handle: 对外暴露的延迟响应句柄。
        step: 待解析的脚本化响应。
        context: 发起请求时的 transcript 上下文。
        options: 发起请求时的 simple stream 选项。
        model: 发起请求的模型。
        pending_fetches: 返回原始 handle 前剩余的空响应次数。
        cancelled: 是否已被取消。
        final: 解析后的最终消息缓存。
    """

    handle: DeferredHandle
    step: FauxResponseStep
    context: TranscriptContext
    options: SimpleStreamOptions | None
    model: Model
    pending_fetches: int
    cancelled: bool = False
    final: AssistantMessage | None = None


# --------------------------------------------------------------------------------------
# 核心
# --------------------------------------------------------------------------------------


class FauxCore:
    """:func:`faux_provider` 与 compat 注册表共用的可脚本化核心。

    Attributes:
        api: provider 的 API 标识。
        provider: provider id。
        min_token_size: delta 分片的最小 token 数。
        max_token_size: delta 分片的最大 token 数。
        tokens_per_second: 模拟的 stream 速率；为 ``None`` 时不限速。
        state: 供测试断言的调用计数器。
        models: 已解析的模型列表。
    """

    def __init__(self, options: RegisterFauxProviderOptions | None = None) -> None:
        """按选项初始化核心并解析模型列表。

        模型定义中的字典型 ``cost`` 会被规约为 :class:`ModelCost`，缺省字段
        使用与默认模型一致的取值。

        Args:
            options: 注册选项；为 ``None`` 时使用默认 API、provider 与单个模型。
        """
        options = options or RegisterFauxProviderOptions()
        self.api = options.api or _random_id(DEFAULT_API)
        self.provider = options.provider or DEFAULT_PROVIDER
        token_size = options.token_size or FauxTokenSize()
        self.min_token_size = max(1, min(token_size.min or DEFAULT_MIN_TOKEN_SIZE, token_size.max or DEFAULT_MAX_TOKEN_SIZE))
        self.max_token_size = max(self.min_token_size, token_size.max or DEFAULT_MAX_TOKEN_SIZE)
        self.tokens_per_second = options.tokens_per_second
        self.state = FauxProviderState()
        self._deferred_options = options.deferred or FauxDeferredOptions()
        self._pending_responses: list[FauxResponseStep] = []
        self._prompt_cache: dict[str, str] = {}
        self._deferred: dict[str, _DeferredEntry] = {}

        definitions = options.models or [
            FauxModelDefinition(
                id=DEFAULT_MODEL_ID,
                name=DEFAULT_MODEL_NAME,
                reasoning=False,
                input=["text"],
                cost=ModelCost(),
                context_window=128000,
                max_tokens=16384,
            )
        ]
        self.models: list[Model] = []
        for definition in definitions:
            cost = definition.cost
            if isinstance(cost, dict):
                cost = from_json(ModelCost, cost)
            self.models.append(
                Model(
                    id=definition.id,
                    name=definition.name or definition.id,
                    api=self.api,
                    provider=self.provider,
                    base_url=DEFAULT_BASE_URL,
                    reasoning=definition.reasoning if definition.reasoning is not None else False,
                    input=definition.input if definition.input is not None else ["text"],
                    input_limits=definition.input_limits,
                    cost=cost if cost is not None else ModelCost(),
                    context_window=definition.context_window if definition.context_window is not None else 128000,
                    max_tokens=definition.max_tokens if definition.max_tokens is not None else 16384,
                )
            )

    async def _resolve_response(
        self,
        step: FauxResponseStep,
        context: TranscriptContext,
        stream_options: SimpleStreamOptions | None,
        request_model: Model,
    ) -> AssistantMessage:
        """解析一条脚本化响应并补齐 usage。

        响应既可以是消息，也可以是以上下文、选项、状态与模型为参数的
        工厂函数（同步或异步）。

        Args:
            step: 队首的脚本化响应。
            context: 本次请求的 transcript 上下文。
            stream_options: 本次请求的 simple stream 选项。
            request_model: 发起请求的模型。

        Returns:
            克隆并绑定到 ``request_model``、usage 已估算的 assistant 消息。
        """
        resolved = await maybe_await(
            step(context, stream_options, self.state, request_model) if callable(step) else step
        )
        return with_usage_estimate(
            clone_message(resolved, self.api, self.provider, request_model.id),
            context,
            stream_options,
            self._prompt_cache,
        )

    def get_model(self, requested_model_id: str | None = None) -> Model | None:
        """第一个模型，或具有所请求 id 的模型。

        Args:
            requested_model_id: 目标模型 id；为空时返回第一个模型。

        Returns:
            匹配的模型；id 不存在时返回 ``None``。
        """
        if not requested_model_id:
            return self.models[0]
        return next((candidate for candidate in self.models if candidate.id == requested_model_id), None)

    # -- stream 输出 ----------------------------------------------------------

    def stream(
        self,
        request_model: Model,
        context: TranscriptContext,
        stream_options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """立即返回 stream，脚本化响应在后台执行。

        每次调用从队列头部取走一条响应并递增 :attr:`state.call_count`；请求
        ``deferred`` 时改为登记延迟响应并先返回占位消息。

        Args:
            request_model: 发起请求的模型。
            context: 本次请求的 transcript 上下文。
            stream_options: simple stream 选项；为 ``None`` 时使用默认值。

        Returns:
            立即返回的 assistant 消息事件 stream，事件由后台任务异步推送。
        """
        outer = create_assistant_message_event_stream()
        step = self._pending_responses.pop(0) if self._pending_responses else None
        self.state.call_count += 1

        async def run() -> None:
            """在后台执行脚本化响应并推送到事件流。"""
            try:
                on_response = stream_options.on_response if stream_options is not None else None
                if on_response is not None:
                    await maybe_await(on_response(ProviderResponse(status=200, headers={}), request_model))
                if step is None:
                    message = create_error_message(
                        RuntimeError("No more faux responses queued"),
                        self.api,
                        self.provider,
                        request_model.id,
                    )
                    message = with_usage_estimate(message, context, stream_options, self._prompt_cache)
                    outer.push(error_event("error", message))
                    outer.end(message)
                    return

                if stream_options is not None and stream_options.deferred:
                    handle = DeferredHandle(
                        provider=request_model.provider,
                        model_id=request_model.id,
                        api=request_model.api,
                        id=_random_id("deferred"),
                        poll_after_ms=self._deferred_options.poll_after_ms,
                    )
                    self._deferred[handle.id] = _DeferredEntry(
                        handle=handle,
                        step=step,
                        context=context,
                        options=stream_options,
                        model=request_model,
                        pending_fetches=max(0, math.floor(self._deferred_options.pending_fetches or 0)),
                    )
                    await stream_with_deltas(
                        outer,
                        create_deferred_message(request_model, handle),
                        self.min_token_size,
                        self.max_token_size,
                        self.tokens_per_second,
                        stream_options.signal,
                    )
                    return

                message = await self._resolve_response(step, context, stream_options, request_model)
                await stream_with_deltas(
                    outer,
                    message,
                    self.min_token_size,
                    self.max_token_size,
                    self.tokens_per_second,
                    stream_options.signal if stream_options is not None else None,
                )
            except Exception as error:  # noqa: BLE001 - 编码进 stream 协议里返回
                message = create_error_message(error, self.api, self.provider, request_model.id)
                outer.push(error_event("error", message))
                outer.end(message)

        task = asyncio.ensure_future(run())
        task.add_done_callback(_swallow)
        return outer

    def stream_simple(
        self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """simple options 入口，委托给 :meth:`stream`。

        Args:
            model: 发起请求的模型。
            context: 本次请求的 transcript 上下文。
            options: simple stream 选项；为 ``None`` 时使用默认值。

        Returns:
            与 :meth:`stream` 相同的事件 stream。
        """
        return self.stream(model, context, options)

    # -- 延迟响应 ---------------------------------------------------------------

    def fetch_deferred(
        self,
        request_model: Model,
        handle: DeferredHandle,
        fetch_options: DeferredFetchOptions | None = None,
    ) -> AssistantMessageEventStream:
        """获取延迟响应，按脚本处理待完成的 pending fetch。

        Args:
            request_model: 发起请求的模型。
            handle: 延迟响应句柄，其 provider/model/api 必须与登记时一致。
            fetch_options: deferred fetch 选项；为 ``None`` 时使用默认值。

        Returns:
            立即返回的事件 stream；未知或已取消的 handle 会以 error 事件收尾。
        """
        outer = create_assistant_message_event_stream()
        self.state.deferred_fetch_count += 1

        async def run() -> None:
            """在后台解析并推送延迟响应结果。"""
            try:
                on_response = fetch_options.on_response if fetch_options is not None else None
                if on_response is not None:
                    await maybe_await(on_response(ProviderResponse(status=200, headers={}), request_model))
                entry = self._deferred.get(handle.id)
                if (
                    entry is None
                    or entry.handle.provider != handle.provider
                    or entry.handle.model_id != handle.model_id
                    or entry.handle.api != handle.api
                ):
                    raise RuntimeError(f"Unknown faux deferred response: {handle.id}")
                if entry.cancelled:
                    raise RuntimeError(f"Faux deferred response was cancelled: {handle.id}")

                if entry.pending_fetches > 0:
                    entry.pending_fetches -= 1
                    await stream_with_deltas(
                        outer,
                        create_deferred_message(request_model, entry.handle),
                        self.min_token_size,
                        self.max_token_size,
                        self.tokens_per_second,
                        fetch_options.signal if fetch_options is not None else None,
                    )
                    return

                if entry.final is None:
                    submission_options = (
                        replace(entry.options, deferred=None, signal=None, on_response=None)
                        if entry.options is not None
                        else None
                    )
                    try:
                        entry.final = await self._resolve_response(
                            entry.step, entry.context, submission_options, entry.model
                        )
                    except Exception as error:  # noqa: BLE001 - 编码进 stream 协议里返回
                        entry.final = create_error_message(error, self.api, self.provider, entry.model.id)
                await stream_with_deltas(
                    outer,
                    entry.final,
                    self.min_token_size,
                    self.max_token_size,
                    self.tokens_per_second,
                    fetch_options.signal if fetch_options is not None else None,
                )
            except Exception as error:  # noqa: BLE001 - 编码进 stream 协议里返回
                message = create_error_message(error, self.api, self.provider, request_model.id)
                outer.push(error_event("error", message))
                outer.end(message)

        task = asyncio.ensure_future(run())
        task.add_done_callback(_swallow)
        return outer

    async def cancel_deferred(
        self,
        request_model: Model,
        handle: DeferredHandle,
        cancel_options: DeferredCancelOptions | None = None,
    ) -> None:
        """把延迟响应标记为已取消，并通知调用方。

        Args:
            request_model: 发起请求的模型。
            handle: 待取消的延迟响应句柄。
            cancel_options: 取消选项，提供可选的回调通知。
        """
        self.state.cancelled_deferred.append(clone(handle))
        entry = self._deferred.get(handle.id)
        if entry is not None:
            entry.cancelled = True
        on_response = cancel_options.on_response if cancel_options is not None else None
        if on_response is not None:
            await maybe_await(on_response(ProviderResponse(status=200, headers={}), request_model))

    # -- 脚本化 -----------------------------------------------------------------

    def set_responses(self, responses: list[FauxResponseStep]) -> None:
        """替换脚本化响应队列。

        Args:
            responses: 新的完整响应队列。
        """
        self._pending_responses = list(responses)

    def append_responses(self, responses: list[FauxResponseStep]) -> None:
        """向脚本化响应队列追加。

        Args:
            responses: 追加到队尾的响应列表。
        """
        self._pending_responses.extend(responses)

    def get_pending_response_count(self) -> int:
        """队列中还剩多少条脚本化响应。

        Returns:
            尚未被消费的响应条数。
        """
        return len(self._pending_responses)


def create_faux_core(options: RegisterFauxProviderOptions | None = None) -> FauxCore:
    """创建可脚本化的 faux 核心。

    Args:
        options: 注册选项；为 ``None`` 时使用默认值。

    Returns:
        新构建的 :class:`FauxCore`。
    """
    return FauxCore(options)


class _FauxAuth:
    """faux provider 的空操作 api-key 认证。"""

    name = "Faux"

    async def resolve(self, **_: Any) -> AuthResult:
        """直接返回空认证，faux provider 不做任何凭证校验。

        Args:
            **_: 认证上下文等参数，全部忽略。

        Returns:
            不含任何凭证信息的 :class:`AuthResult`。
        """
        return AuthResult(auth=ModelAuth())


def faux_provider(options: RegisterFauxProviderOptions | None = None) -> FauxProviderHandle:
    """基于显式 ``Models`` 集合、供测试使用的 faux provider。

    Args:
        options: 注册选项；为 ``None`` 时使用默认 API、provider 与单个模型。

    Returns:
        绑定 provider 与脚本化接口的 :class:`FauxProviderHandle`。

    Examples:
        .. code-block:: python

            faux = faux_provider()
            models = create_models()
            models.set_provider(faux.provider)
            faux.set_responses([faux_assistant_message("hi")])
    """
    core = create_faux_core(options)
    provider = create_provider(
        CreateProviderOptions(
            id=core.provider,
            auth=ProviderAuth(api_key=_FauxAuth()),
            models=core.models,
            api=core,
        )
    )
    return FauxProviderHandle(
        provider=provider,
        api=core.api,
        models=core.models,
        get_model=core.get_model,
        state=core.state,
        set_responses=core.set_responses,
        append_responses=core.append_responses,
        get_pending_response_count=core.get_pending_response_count,
    )
