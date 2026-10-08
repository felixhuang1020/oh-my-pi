"""OpenAI Responses API adapter 的共享机制。

移植自 ``packages/ai/src/api/openai-responses-shared.ts``。本模块持有
:mod:`pi_ai.api.openai_responses` 使用的共享部分：Responses 输入转换、
工具转换以及 stream 事件状态机。

TypeScript 原版通过 ``openai`` SDK 访问 Responses API；移植版则直接经
:mod:`httpx` 走 HTTP（见 :func:`send_streaming_request` 和
:func:`iter_stream_payloads`），因此这里解析出的线上对象是普通 ``dict``。
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.parse
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..models import calculate_cost
from ..types import (
    AssistantMessage,
    Model,
    StopReason,
    SystemMessage,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    TranscriptContext,
    Usage,
)
from ..utils.abort import AbortSignal
from ..utils.async_utils import maybe_await
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.events import (
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
from ..utils.hash import short_hash
from ..utils.http import iter_response_lines, read_error_body
from ..utils.json_parse import parse_streaming_json
from ..utils.sanitize_unicode import sanitize_surrogates
from ..utils.serde import to_json
from ..utils.sse import iter_sse_events
from ..utils.text import get_system_message_text, render_system_message_update
from ..utils.transcript import resolve_transcript, resolve_transcript_tools
from .constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    get_grammar_tool_input,
    get_json_schema_tool_parameters,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from .transform_messages import transform_messages

__all__ = [
    "UNSET",
    "ConvertResponsesMessagesOptions",
    "ConvertResponsesToolsOptions",
    "OpenAIResponsesStreamOptions",
    "ProviderHTTPError",
    "convert_responses_messages",
    "convert_responses_tools",
    "convert_tool_result_output",
    "encode_text_signature_v1",
    "iter_stream_payloads",
    "iter_with_abort",
    "join_url",
    "map_stop_reason",
    "parse_text_signature",
    "process_responses_stream",
    "send_streaming_request",
]

#: 与 JavaScript 紧凑版 ``JSON.stringify`` 输出一致的 JSON 分隔符。
_JSON_SEPARATORS = (",", ":")


def _json_dumps(value: Any) -> str:
    """与 ``JSON.stringify`` 完全一致地序列化：紧凑、不做 ASCII 转义。

    Args:
        value: 任意可 JSON 序列化的值。

    Returns:
        使用紧凑分隔符、保留非 ASCII 字符的 JSON 字符串。
    """
    return json.dumps(value, ensure_ascii=False, separators=_JSON_SEPARATORS, default=str)


def _parse_json_or_none(text: str) -> Any:
    """把文本解析为 JSON；空文本或解析失败时返回 ``None``。

    Args:
        text: 待解析的文本。

    Returns:
        解析结果；无法解析时为 ``None``。
    """
    try:
        return json.loads(text) if text.strip() else None
    except ValueError:
        return None


def _js_truthy(value: Any) -> bool:
    """针对 provider 错误响应体中出现的值，实现 JavaScript 的真值判定。

    Args:
        value: 待判定的值。

    Returns:
        按 JavaScript 语义为真时返回 ``True``。
    """
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return len(value) > 0
    if isinstance(value, (int, float)):
        return value != 0
    return True


# =============================================================================
# HTTP 管道（移植版对 SDK 流式请求的替代实现）
# =============================================================================


class ProviderHTTPError(Exception):
    """非 2xx 的 provider 响应，形状仿照 SDK 的 ``APIError``。

    对齐 ``openai`` 的 ``APIError``：:attr:`error` 是解包后的 ``body["error"]``
    值（或 ``None``），消息遵循 ``APIError.makeMessage``。该形状使得
    :func:`pi_ai.utils.provider_retry.retry_provider_request` 和
    :func:`pi_ai.utils.error_body.normalize_provider_error` 可以原样消费它。

    Attributes:
        status: HTTP 状态码。
        headers: 响应头。
        error: 解包后的 ``body["error"]`` 值；缺失时为 ``None``。
    """

    def __init__(self, status: int, headers: Any, error_response: Any) -> None:
        """初始化 provider HTTP 错误。

        Args:
            status: HTTP 状态码。
            headers: 响应头。
            error_response: 解析后的错误响应体，通常形如 ``{"error": ...}``。
        """
        inner = error_response.get("error") if isinstance(error_response, dict) else None
        super().__init__(_make_api_error_message(status, inner))
        self.status = status
        self.headers = headers
        self.error = inner


def _make_api_error_message(status: int | None, inner: Any) -> str:
    """移植 ``openai`` 的 ``APIError.makeMessage``。

    Args:
        status: HTTP 状态码；缺失时为 ``None``。
        inner: 解包后的 ``error`` 值。

    Returns:
        组合 ``<status> <message>`` 的展示文本。
    """
    message_value = inner.get("message") if isinstance(inner, dict) else None
    if _js_truthy(message_value):
        message = (
            message_value if isinstance(message_value, str) else _json_dumps(message_value)
        )
    elif _js_truthy(inner):
        message = _json_dumps(inner)
    else:
        message = None
    if status and message:
        return f"{status} {message}"
    if status:
        return f"{status} status code (no body)"
    if message:
        return message
    return "(no status code or body)"


async def _await_with_abort(awaitable: Any, signal: AbortSignal | None) -> Any:
    """等待 ``awaitable``，``signal`` 一中止便取消它。

    Args:
        awaitable: 待等待的 awaitable 或 future。
        signal: 中止信号；为 ``None`` 时直接等待。

    Returns:
        ``awaitable`` 的结果。

    Raises:
        BaseException: 中止信号对应的异常，或 ``awaitable`` 自身抛出的异常。
    """
    if signal is None:
        return await awaitable
    if signal.aborted:
        task = asyncio.ensure_future(awaitable)
        task.add_done_callback(_swallow)
        raise signal.exception()

    task = asyncio.ensure_future(awaitable)
    abort_task = asyncio.ensure_future(signal.wait())
    try:
        done, _ = await asyncio.wait({task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        task.cancel()
        abort_task.cancel()
        raise
    if task in done:
        abort_task.cancel()
        return task.result()
    task.cancel()
    task.add_done_callback(_swallow)
    raise signal.exception()


def _swallow(task: asyncio.Future[Any]) -> None:
    """取出已完成 task 的异常，避免被报告为未处理异常。

    Args:
        task: 已完成或已取消的 asyncio task。
    """
    if task.cancelled() or not task.done():
        return
    task.exception()


async def _safe_anext(iterator: AsyncIterator[Any]) -> tuple[bool, Any]:
    """读取异步迭代器的下一项，把结束表示为 ``(False, None)``。

    Args:
        iterator: 目标异步迭代器。

    Returns:
        二元组 ``(has_value, value)``；迭代结束时为 ``(False, None)``。
    """
    try:
        return True, await iterator.__anext__()
    except StopAsyncIteration:
        return False, None


async def iter_with_abort(iterable: Any, signal: AbortSignal | None) -> AsyncIterator[Any]:
    """迭代 ``iterable``，``signal`` 中止时停止（并取消进行中的读取）。

    Args:
        iterable: 任意异步可迭代对象。
        signal: 中止信号；为 ``None`` 时退化为普通迭代。

    Yields:
        底层迭代器逐个产出的值。

    Raises:
        BaseException: 中止信号对应的异常。
    """
    iterator = iterable.__aiter__()
    while True:
        if signal is None:
            has_value, value = await _safe_anext(iterator)
            if not has_value:
                return
            yield value
            continue
        if signal.aborted:
            raise signal.exception()

        next_task = asyncio.ensure_future(_safe_anext(iterator))
        abort_task = asyncio.ensure_future(signal.wait())
        try:
            done, _ = await asyncio.wait({next_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            next_task.cancel()
            abort_task.cancel()
            raise
        if next_task in done:
            abort_task.cancel()
            has_value, value = next_task.result()
            if not has_value:
                return
            yield value
        else:
            next_task.cancel()
            next_task.add_done_callback(_swallow)
            raise signal.exception()


def join_url(base_url: str, path: str, query: Mapping[str, str] | None = None) -> str:
    """把 ``path`` 拼接到 ``base_url``，保留并扩展已有查询串。

    Args:
        base_url: 基础 URL，可自带查询串。
        path: 追加到基础路径之后的路径片段。
        query: 需要并入的查询参数。

    Returns:
        拼接后的完整 URL。
    """
    parts = urllib.parse.urlsplit(base_url.strip())
    base_path = parts.path.rstrip("/")
    query_parts = [parts.query] if parts.query else []
    if query:
        query_parts.append(urllib.parse.urlencode(query))
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, f"{base_path}{path}", "&".join(query_parts), parts.fragment)
    )


async def send_streaming_request(
    client: httpx.AsyncClient,
    url: str,
    params: Mapping[str, Any],
    signal: AbortSignal | None = None,
) -> httpx.Response:
    """发起 POST 流式请求，响应头到达后即返回。

    非 2xx 响应会被读空、关闭并抛出 :class:`ProviderHTTPError`，以便调用方
    （通常位于 :func:`retry_provider_request` 内部）决定是否重试。

    Args:
        client: 复用的 httpx 异步客户端。
        url: 目标端点 URL。
        params: 请求体参数，会被序列化为 JSON。
        signal: 中止信号；为 ``None`` 时不响应中止。

    Returns:
        已建立流式读取的 httpx 响应。

    Raises:
        ProviderHTTPError: 当响应状态码不小于 400 时。
    """
    request = client.build_request("POST", url, json=to_json(dict(params)))
    response = await _await_with_abort(client.send(request, stream=True), signal)
    if response.status_code >= 400:
        body = await read_error_body(response)
        headers = httpx.Headers(response.headers)
        await response.aclose()
        raise ProviderHTTPError(response.status_code, headers, _parse_json_or_none(body))
    return response


async def iter_stream_payloads(
    response: httpx.Response,
    signal: AbortSignal | None = None,
) -> AsyncIterator[Any]:
    """解码流式 provider 响应中的 SSE ``data:`` 负载。

    Args:
        response: 处于流式读取状态的 httpx 响应。
        signal: 中止信号；为 ``None`` 时不响应中止。

    Yields:
        逐个解析出的 SSE ``data:`` JSON 负载；``[DONE]`` 会结束迭代。

    Raises:
        json.JSONDecodeError: 当某条负载不是合法 JSON 时。
    """
    events = iter_sse_events(iter_response_lines(response))
    async for event in iter_with_abort(events, signal):
        data = event.data.strip()
        if not data:
            continue
        if data == "[DONE]":
            return
        yield json.loads(data)


# =============================================================================
# 工具函数
# =============================================================================


def encode_text_signature_v1(id: str, phase: str | None = None) -> str:
    """编码存储在 :class:`TextContent` 上的 Responses message-id 签名。

    Args:
        id: Responses message id。
        phase: message 阶段（``"commentary"``/``"final_answer"``）；为 ``None`` 时不写入。

    Returns:
        形如 ``{"v": 1, "id": ...}`` 的 JSON 字符串。
    """
    payload: dict[str, Any] = {"v": 1, "id": id}
    if phase:
        payload["phase"] = phase
    return _json_dumps(payload)


def parse_text_signature(signature: str | None) -> dict[str, Any] | None:
    """把文本签名解码为 ``{"id", "phase"?}``，兼容遗留的纯 id 形式。

    Args:
        signature: 待解码的签名字符串。

    Returns:
        含 ``id`` 及可选 ``phase`` 的字典；``signature`` 为空时返回 ``None``。
    """
    if not signature:
        return None
    if signature.startswith("{"):
        try:
            parsed = json.loads(signature)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("v") == 1 and isinstance(parsed.get("id"), str):
            if parsed.get("phase") in ("commentary", "final_answer"):
                return {"id": parsed["id"], "phase": parsed["phase"]}
            return {"id": parsed["id"]}
    return {"id": signature}


def convert_tool_result_output(
    content: Sequence[TextContent],
) -> Any:
    """把工具结果内容转换为 Responses 的 ``function_call_output`` 负载。

    Args:
        content: 工具结果内容块序列。

    Returns:
        纯文本负载；内容为空时返回占位文本。
    """
    text_result = "\n".join(block.text for block in content if getattr(block, "type", None) == "text")
    return sanitize_surrogates(text_result) if text_result else sanitize_surrogates("(no tool output)")


# =============================================================================
# 选项
# =============================================================================


class _Unset:
    """区分“调用方未指定该项”与显式的 JSON ``null``。

    上游的 option 对象是普通 JavaScript 对象，``undefined`` 与 ``null`` 是不同的值。
    ``convertResponsesTools`` 把缺省的 ``strict`` 当作 ``false`` 处理，但会原样转发
    显式的 ``null``；因此 Python 里用 ``None`` 做默认值会把两者混为一谈。
    """

    _instance: "_Unset | None" = None

    def __new__(cls) -> "_Unset":
        """返回进程内唯一的 ``_Unset`` 单例。

        Returns:
            单例实例。
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        """返回便于调试的 ``"UNSET"`` 文本。

        Returns:
            固定字符串 ``"UNSET"``。
        """
        return "UNSET"

    def __bool__(self) -> bool:
        """让 ``UNSET`` 在布尔上下文中始终为假。

        Returns:
            固定为 ``False``。
        """
        return False


#: 表示“调用方未指定该选项”的标记。
UNSET = _Unset()


@dataclass
class ConvertResponsesToolsOptions:
    """各 Responses adapter 共用的工具转换开关。

    Attributes:
        strict: ``strict`` 字段取值；默认 ``UNSET`` 表示调用方未指定。
        supports_strict_mode: 模型是否支持 strict schema。
        supports_openai_grammar_tools: 模型是否支持 OpenAI 语法约束工具。
        tool_search_result: 工具是否作为 tool search 结果延迟加载。
    """

    strict: bool | None | _Unset = UNSET
    supports_strict_mode: bool | None = field(default=None, metadata={"alias": "supportsStrictMode"})
    supports_openai_grammar_tools: bool | None = field(
        default=None, metadata={"alias": "supportsOpenAIGrammarTools"}
    )
    tool_search_result: bool | None = field(default=None, metadata={"alias": "toolSearchResult"})


@dataclass
class ConvertResponsesMessagesOptions:
    """:func:`convert_responses_messages` 的选项。

    Attributes:
        include_system_prompt: 是否携带开头的 system prompt；默认视为 ``True``。
        grammar_tool_input_properties: 语法约束工具的输入属性映射。
        supports_mid_convo_system_messages: 后续 system 消息是否原位发送。
        supports_additional_tools: 是否支持 ``additional_tools`` 项。
        supports_tool_search: 是否支持 tool search 形式的工具追加。
        tool_options: 传给 :func:`convert_responses_tools` 的选项。
    """

    include_system_prompt: bool | None = field(default=None, metadata={"alias": "includeSystemPrompt"})
    grammar_tool_input_properties: Mapping[str, str] | None = field(
        default=None, metadata={"alias": "grammarToolInputProperties"}
    )
    #: 后续 system 消息是否原位发送；否则会并入开头的 prompt。
    supports_mid_convo_system_messages: bool | None = field(
        default=None, metadata={"alias": "supportsMidConvoSystemMessages"}
    )
    supports_additional_tools: bool | None = field(default=None, metadata={"alias": "supportsAdditionalTools"})
    supports_tool_search: bool | None = field(default=None, metadata={"alias": "supportsToolSearch"})
    tool_options: ConvertResponsesToolsOptions | None = field(default=None, metadata={"alias": "toolOptions"})


@dataclass
class OpenAIResponsesStreamOptions:
    """:func:`process_responses_stream` 的选项。

    Attributes:
        on_provider_stream_event: 每个原始 provider 事件的可选回调。
        service_tier: 请求时使用的服务档位。
        grammar_tool_input_properties: 语法约束工具的输入属性映射。
        resolve_service_tier: 用响应与请求档位推导最终档位的可选函数。
        apply_service_tier_pricing: 按最终档位调整计费的可选函数。
    """

    on_provider_stream_event: Callable[..., Any] | None = field(
        default=None, metadata={"alias": "onProviderStreamEvent"}
    )
    service_tier: str | None = field(default=None, metadata={"alias": "serviceTier"})
    grammar_tool_input_properties: Mapping[str, str] | None = field(
        default=None, metadata={"alias": "grammarToolInputProperties"}
    )
    resolve_service_tier: Callable[..., Any] | None = field(
        default=None, metadata={"alias": "resolveServiceTier"}
    )
    apply_service_tier_pricing: Callable[..., Any] | None = field(
        default=None, metadata={"alias": "applyServiceTierPricing"}
    )


# =============================================================================
# 消息转换
# =============================================================================


def convert_responses_messages(
    model: Model,
    context: TranscriptContext,
    allowed_tool_call_providers: Iterable[str],
    options: ConvertResponsesMessagesOptions | None = None,
) -> list[dict[str, Any]]:
    """把 pi 转录（transcript）转换为 Responses API 的 ``input`` 数组。

    Args:
        model: 目标模型元数据。
        context: 待转换的对话上下文。
        allowed_tool_call_providers: 允许沿用原始 tool call id 的 provider 集合。
        options: 消息转换选项；为 ``None`` 时使用默认值。

    Returns:
        Responses API 的 ``input`` 数组。
    """
    options = options or ConvertResponsesMessagesOptions()
    normalized_context = resolve_transcript(context, options.supports_mid_convo_system_messages)
    messages: list[dict[str, Any]] = []
    allowed_providers = set(allowed_tool_call_providers)

    def normalize_id_part(part: str) -> str:
        """把 id 片段清洗为只含 ``[a-zA-Z0-9_-]`` 且不超过 64 字符。

        Args:
            part: 原始 id 片段。

        Returns:
            规范化后的 id 片段。
        """
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", part)
        normalized = sanitized[:64] if len(sanitized) > 64 else sanitized
        return re.sub(r"_+$", "", normalized)

    def build_foreign_responses_item_id(item_id: str) -> str:
        """为来自其他 provider 的 tool call 生成稳定的 ``fc_`` 前缀 id。

        Args:
            item_id: 原始 item id。

        Returns:
            截断到 64 字符以内的新 item id。
        """
        normalized = f"fc_{short_hash(item_id)}"
        return normalized[:64] if len(normalized) > 64 else normalized

    def normalize_tool_call_id(id: str, _target_model: Model, source: AssistantMessage) -> str:
        """规范化 ``call_id|item_id`` 形式的工具调用 id。

        Args:
            id: 原始工具调用 id。
            _target_model: :func:`transform_messages` 传入的目标模型（此处未使用）。
            source: 产生该工具调用的 assistant 消息。

        Returns:
            规范化后的 ``call_id|item_id`` 字符串。
        """
        if model.provider not in allowed_providers:
            return normalize_id_part(id)
        if "|" not in id:
            return normalize_id_part(id)
        parts = id.split("|")
        call_id = parts[0]
        item_id = parts[1]
        normalized_call_id = normalize_id_part(call_id)
        is_foreign_tool_call = source.provider != model.provider or source.api != model.api
        normalized_item_id = (
            build_foreign_responses_item_id(item_id) if is_foreign_tool_call else normalize_id_part(item_id)
        )
        # OpenAI Responses API 要求 item id 以 "fc" 开头
        if not normalized_item_id.startswith("fc_"):
            normalized_item_id = normalize_id_part(f"fc_{normalized_item_id}")
        return f"{normalized_call_id}|{normalized_item_id}"

    transformed_messages = transform_messages(normalized_context.messages, model, normalize_tool_call_id)
    transcript_tools = resolve_transcript_tools(
        normalized_context.messages,
        bool(options.supports_additional_tools) or bool(options.supports_tool_search),
    )

    def append_system_tool_additions(message: SystemMessage, seed: str) -> None:
        """把 system 消息新增的工具追加为 Responses 输入项。

        按模型能力选择 ``additional_tools`` 或 tool search 形式；两者都不支持时静默跳过。

        Args:
            message: 携带 ``tools_added`` 的 system 消息。
            seed: 用于生成确定性 call id 的种子字符串。
        """
        tools = (message.tools_added or []) if transcript_tools.anchors_additions else []
        if len(tools) == 0:
            return
        if options.supports_additional_tools:
            messages.append(
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": convert_responses_tools(tools, options.tool_options),
                }
            )
            return
        if not options.supports_tool_search:
            return
        names = [tool.name for tool in tools]
        seed_key = f"{seed}:{','.join(names)}"
        call_id = f"pi_tool_load_{short_hash(seed_key)}"
        messages.append(
            {
                "type": "tool_search_call",
                "call_id": call_id,
                "execution": "client",
                "status": "completed",
                "arguments": {"query": " ".join(names), "limit": len(names)},
            }
        )
        messages.append(
            {
                "type": "tool_search_output",
                "call_id": call_id,
                "execution": "client",
                "status": "completed",
                "tools": convert_responses_tools(
                    tools, _replace_tool_options(options.tool_options, tool_search_result=True)
                ),
            }
        )

    include_initial_system_message = (
        options.include_system_prompt if options.include_system_prompt is not None else True
    )
    compat = model.compat
    instruction_role = (
        "developer"
        if model.reasoning and (compat is None or compat.supports_developer_role is not False)
        else "system"
    )

    msg_index = 0
    source_index = 0
    for msg in transformed_messages:
        is_leading_system_message = source_index == 0 and msg.role == "system"
        source_index += 1
        if msg.role == "system":
            if not is_leading_system_message:
                append_system_tool_additions(msg, f"system:{msg_index}")
            if (not is_leading_system_message) or include_initial_system_message:
                text = (
                    get_system_message_text(msg)
                    if is_leading_system_message
                    else render_system_message_update(msg)
                )
                if len(text) > 0:
                    messages.append({"role": instruction_role, "content": sanitize_surrogates(text)})
        elif msg.role == "user":
            if isinstance(msg.content, str):
                messages.append(
                    {"role": "user", "content": [{"type": "input_text", "text": sanitize_surrogates(msg.content)}]}
                )
            else:
                content: list[dict[str, Any]] = [
                    {"type": "input_text", "text": sanitize_surrogates(item.text)}
                    for item in msg.content
                    if getattr(item, "type", None) == "text"
                ]
                if len(content) == 0:
                    continue
                messages.append({"role": "user", "content": content})
        elif msg.role == "assistant":
            output: list[dict[str, Any]] = []
            assistant_msg = msg
            is_same_provider_and_api = (
                assistant_msg.provider == model.provider and assistant_msg.api == model.api
            )
            is_same_model = is_same_provider_and_api and assistant_msg.model == model.id
            is_different_model = is_same_provider_and_api and assistant_msg.model != model.id
            text_block_index = 0

            for block in msg.content:
                block_type = getattr(block, "type", None)
                if block_type == "thinking":
                    if block.thinking_signature:
                        output.append(json.loads(block.thinking_signature))
                elif block_type == "text":
                    text_block = block
                    parsed_signature = parse_text_signature(text_block.text_signature)
                    fallback_message_id = (
                        f"msg_pi_{msg_index}" if text_block_index == 0 else f"msg_pi_{msg_index}_{text_block_index}"
                    )
                    text_block_index += 1
                    msg_id = parsed_signature.get("id") if parsed_signature else None
                    if not msg_id:
                        msg_id = fallback_message_id
                    elif len(msg_id) > 64:
                        msg_id = f"msg_{short_hash(msg_id)}"
                    item: dict[str, Any] = {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": sanitize_surrogates(text_block.text),
                                "annotations": [],
                            }
                        ],
                        "status": "completed",
                        "id": msg_id,
                    }
                    if parsed_signature and parsed_signature.get("phase") is not None:
                        item["phase"] = parsed_signature["phase"]
                    output.append(item)
                elif block_type == "toolCall":
                    tool_call = block
                    parts = tool_call.id.split("|")
                    call_id = parts[0]
                    item_id_raw = parts[1] if len(parts) > 1 else None
                    grammar_properties = options.grammar_tool_input_properties or {}
                    custom_input_property = grammar_properties.get(tool_call.name)
                    item_id: str | None = item_id_raw

                    # 不同模型的消息直接丢弃 id 以避免配对校验。同时丢弃与重放 item 类型
                    # 不符的 id：function_call 的 id 必须是 fc_*，custom_tool_call 的 id
                    # 必须是 ctc_*。
                    item_id_prefix = "fc_" if custom_input_property is None else "ctc_"
                    if is_different_model or item_id is None or not item_id.startswith(item_id_prefix):
                        item_id = None

                    namespace = (
                        {"namespace": tool_call.namespace}
                        if is_same_model and tool_call.namespace is not None
                        else {}
                    )
                    if custom_input_property is not None:
                        entry: dict[str, Any] = {"type": "custom_tool_call"}
                        if item_id is not None:
                            entry["id"] = item_id
                        entry["call_id"] = call_id
                        entry["name"] = tool_call.name
                        entry["input"] = sanitize_surrogates(
                            get_grammar_tool_input(tool_call.name, tool_call.arguments, custom_input_property)
                        )
                        entry.update(namespace)
                        output.append(entry)
                    else:
                        entry = {"type": "function_call"}
                        if item_id is not None:
                            entry["id"] = item_id
                        entry["call_id"] = call_id
                        entry["name"] = tool_call.name
                        entry["arguments"] = _json_dumps(tool_call.arguments)
                        entry.update(namespace)
                        output.append(entry)
            if len(output) == 0:
                continue
            messages.extend(output)
        elif msg.role == "toolResult":
            call_id = msg.tool_call_id.split("|")[0]
            output_value = convert_tool_result_output(msg.content)
            grammar_properties = options.grammar_tool_input_properties or {}
            if msg.tool_name in grammar_properties:
                messages.append({"type": "custom_tool_call_output", "call_id": call_id, "output": output_value})
            else:
                messages.append({"type": "function_call_output", "call_id": call_id, "output": output_value})
        if not is_leading_system_message:
            msg_index += 1

    return messages


def _replace_tool_options(
    options: ConvertResponsesToolsOptions | None,
    **overrides: Any,
) -> ConvertResponsesToolsOptions:
    """复制工具转换选项并覆盖指定字段。

    Args:
        options: 原始选项；为 ``None`` 时使用默认值。
        **overrides: 需要覆盖的字段。

    Returns:
        覆盖后的新选项对象。
    """
    base = options or ConvertResponsesToolsOptions()
    values = {
        "strict": base.strict,
        "supports_strict_mode": base.supports_strict_mode,
        "supports_openai_grammar_tools": base.supports_openai_grammar_tools,
        "tool_search_result": base.tool_search_result,
    }
    values.update(overrides)
    return ConvertResponsesToolsOptions(**values)


# =============================================================================
# 工具转换
# =============================================================================


def convert_responses_tools(
    tools: Sequence[Tool],
    options: ConvertResponsesToolsOptions | None = None,
) -> list[dict[str, Any]]:
    """把 pi 工具声明转换为 Responses API 的工具对象。

    语法约束工具转换为 ``custom`` 形状，其余转换为 ``function`` 形状；
    ``strict`` 缺省时按 ``false`` 处理，但显式 ``null`` 会原样转发。

    Args:
        tools: 待转换的工具声明序列。
        options: 转换开关；为 ``None`` 时全部使用默认值。

    Returns:
        Responses API 的工具对象列表。
    """
    # `undefined`（未指定）默认视为 false；显式 `null` 则原样转发。
    default_strict = False if options is None or options.strict is UNSET else options.strict
    supports_strict_mode = (
        options.supports_strict_mode
        if options is not None and options.supports_strict_mode is not None
        else True
    )
    supports_openai_grammar_tools = bool(options.supports_openai_grammar_tools) if options else False

    converted: list[dict[str, Any]] = []
    for tool in tools:
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar:
            entry: dict[str, Any] = {
                "type": "custom",
                "name": tool.name,
                "description": tool.description,
                "format": {
                    "type": "grammar",
                    "syntax": grammar.format,
                    "definition": grammar.definition,
                },
            }
            if options and options.tool_search_result:
                entry["defer_loading"] = True
            converted.append(entry)
            continue

        constrained_strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        strict = constrained_strict if constrained_strict is not None else default_strict
        function_tool: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": get_json_schema_tool_parameters(tool, strict is True),
        }
        if options and options.tool_search_result:
            function_tool["defer_loading"] = True
        if supports_strict_mode:
            function_tool["strict"] = strict
        converted.append(function_tool)
    return converted


# =============================================================================
# stream 处理
# =============================================================================


@dataclass
class _ResponsesOutputSlot:
    """一个流式输出块及其事件流引用的索引。

    Attributes:
        type: 槽位类型（``"thinking"``/``"text"``/``"toolCall"``）。
        block: 累积中的内容块对象。
        content_index: 该块在 ``output.content`` 中的下标。
    """

    type: str
    block: Any
    content_index: int


@dataclass
class _MappedStopReason:
    """``status`` 到 pi stop reason 的映射结果。

    Attributes:
        stop_reason: 映射后的 stop reason。
        error_message: 需要上报的错误消息；无错误时为 ``None``。
    """

    stop_reason: StopReason | str
    error_message: str | None = None


def _get_custom_tool_call_input(block: ToolCall) -> str:
    """读取语法约束工具调用当前累积的输入文本。

    Args:
        block: 目标工具调用块。

    Returns:
        已累积的输入字符串；该调用不受语法约束时为空串。
    """
    custom_input = getattr(block, "custom_input", None)
    property_name = custom_input.get("property") if custom_input else None
    if property_name is None:
        return ""
    value = block.arguments.get(property_name)
    return value if isinstance(value, str) else ""


def _append_custom_tool_call_input(block: ToolCall, next_input: str, close: bool) -> str | None:
    """累积语法约束工具调用的输入增量。

    Args:
        block: 目标工具调用块。
        next_input: 本次追加的完整输入文本。
        close: 是否已收到完成事件，需要冲掉缓冲区。

    Returns:
        需要推送的增量文本；该调用不受语法约束时为 ``None``。
    """
    custom_input = getattr(block, "custom_input", None)
    if not custom_input:
        return None
    delta = append_grammar_tool_input_json_delta(
        custom_input["json_buffer"], custom_input["property"], next_input, close
    )
    block.arguments = {custom_input["property"]: next_input}
    return delta


async def process_responses_stream(
    openai_stream: AsyncIterator[dict[str, Any]],
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    options: OpenAIResponsesStreamOptions | None = None,
) -> None:
    """把 Responses 事件流折叠进 ``output``，同时推送 stream 事件。

    Args:
        openai_stream: 已解码的 Responses 事件流。
        output: 正在累积的 assistant 消息。
        stream: 对外推送事件的 stream。
        model: 目标模型元数据，会传给 provider stream 事件回调。
        options: 流处理选项；为 ``None`` 时使用默认行为。

    Raises:
        RuntimeError: 收到 ``error``/``response.failed`` 事件、缺少终止事件，
            或结束时仍有未完成的工具调用时。
    """
    saw_terminal_response_event = False
    output_slots: dict[Any, _ResponsesOutputSlot] = {}
    reasoning_blocks_by_id: dict[str, ThinkingContent] = {}

    def apply_message_phase_stop_reason(item: Mapping[str, Any]) -> None:
        """当 message 标记为 ``final_answer`` 时把 stop reason 置为 ``"stop"``。

        Args:
            item: Responses 输出项。
        """
        if item.get("type") == "message" and item.get("phase") == "final_answer":
            output.stop_reason = "stop"

    def get_slot(output_index: Any, slot_type: str) -> _ResponsesOutputSlot | None:
        """按 ``output_index`` 取槽位，并校验类型一致。

        Args:
            output_index: Responses 事件的输出下标。
            slot_type: 期望的槽位类型。

        Returns:
            匹配的槽位；不存在或类型不符时为 ``None``。
        """
        slot = output_slots.get(output_index)
        return slot if slot is not None and slot.type == slot_type else None

    def push_tool_call_delta(slot: _ResponsesOutputSlot, delta: str | None) -> None:
        """推送工具调用参数增量事件。

        Args:
            slot: 目标工具调用槽位。
            delta: 增量文本；为 ``None`` 时不推送。
        """
        if delta is None:
            return
        stream.push(tool_call_delta_event(slot.content_index, delta, output))

    def create_slot(output_index: Any, item: Mapping[str, Any]) -> _ResponsesOutputSlot | None:
        """为新的输出项创建槽位并推送对应的 start 事件。

        仅处理 ``reasoning``、``message``、``function_call``、
        ``custom_tool_call`` 四类输出项。

        Args:
            output_index: Responses 事件的输出下标。
            item: Responses 输出项。

        Returns:
            新建的槽位；输出项类型不受支持时为 ``None``。
        """
        item_type = item.get("type")
        if item_type == "reasoning":
            block = ThinkingContent(type="thinking", thinking="")
            output.content.append(block)
            slot = _ResponsesOutputSlot(type="thinking", block=block, content_index=len(output.content) - 1)
            output_slots[output_index] = slot
            stream.push(thinking_start_event(slot.content_index, output))
            return slot
        if item_type == "message":
            apply_message_phase_stop_reason(item)
            text_block = TextContent(type="text", text="")
            output.content.append(text_block)
            slot = _ResponsesOutputSlot(type="text", block=text_block, content_index=len(output.content) - 1)
            output_slots[output_index] = slot
            stream.push(text_start_event(slot.content_index, output))
            return slot
        if item_type == "function_call":
            block = ToolCall(
                type="toolCall",
                id=f"{item.get('call_id')}|{item.get('id')}",
                name=item.get("name") or "",
                arguments={},
            )
            if item.get("namespace") is not None:
                block.namespace = item["namespace"]
            block.partial_json = item.get("arguments") or ""
            output.content.append(block)
            slot = _ResponsesOutputSlot(type="toolCall", block=block, content_index=len(output.content) - 1)
            output_slots[output_index] = slot
            stream.push(tool_call_start_event(slot.content_index, output))
            return slot
        if item_type == "custom_tool_call":
            grammar_properties = (options.grammar_tool_input_properties if options else None) or {}
            input_property = grammar_properties.get(item.get("name")) or "input"
            input_value = item.get("input") or ""
            block = ToolCall(
                type="toolCall",
                id=f"{item.get('call_id')}|{item.get('id')}",
                name=item.get("name") or "",
                arguments={input_property: input_value},
            )
            if item.get("namespace") is not None:
                block.namespace = item["namespace"]
            block.custom_input = {
                "property": input_property,
                "json_buffer": GrammarToolInputJsonBuffer(),
            }
            output.content.append(block)
            slot = _ResponsesOutputSlot(type="toolCall", block=block, content_index=len(output.content) - 1)
            output_slots[output_index] = slot
            stream.push(tool_call_start_event(slot.content_index, output))
            return slot
        return None

    def get_or_create_slot(output_index: Any, item: Mapping[str, Any]) -> _ResponsesOutputSlot | None:
        """取出已有槽位，不存在时按输出项创建。

        Args:
            output_index: Responses 事件的输出下标。
            item: Responses 输出项。

        Returns:
            已有或新建的槽位；类型不受支持时为 ``None``。
        """
        return output_slots.get(output_index) or create_slot(output_index, item)

    def backfill_reasoning_signatures(response_output: Sequence[Mapping[str, Any]]) -> None:
        """用最终响应中的 ``encrypted_content`` 回填推理块签名。

        流式增量事件不带加密内容，只有终止响应才携带，因此需要在这时补齐，
        否则重放历史时无法还原 reasoning item。

        Args:
            response_output: 最终响应的 ``output`` 数组。
        """
        for item in response_output:
            if item.get("type") != "reasoning" or not item.get("encrypted_content"):
                continue
            block = reasoning_blocks_by_id.get(item.get("id"))
            if block is None or not block.thinking_signature:
                continue
            stored_item = json.loads(block.thinking_signature)
            if stored_item.get("encrypted_content"):
                continue
            block.thinking_signature = _json_dumps({**stored_item, "encrypted_content": item["encrypted_content"]})

    def finalize_response(response: Mapping[str, Any]) -> None:
        """处理终止响应：回填签名、结算用量与费用并确定 stop reason。

        Args:
            response: ``response.completed``/``response.incomplete`` 事件中的响应对象。
        """
        nonlocal saw_terminal_response_event
        saw_terminal_response_event = True
        backfill_reasoning_signatures(response.get("output") or [])
        if response.get("id"):
            output.response_id = response["id"]
        usage = response.get("usage")
        if usage:
            input_details = usage.get("input_tokens_details") or {}
            cached_tokens = input_details.get("cached_tokens") or 0
            cache_write_tokens = input_details.get("cache_write_tokens") or 0
            output_tokens_details = usage.get("output_tokens_details") or {}
            output.usage = Usage(
                # OpenAI 把缓存读与缓存写 token 都计入 input_tokens，因此两者都要减去。
                input=max(0, (usage.get("input_tokens") or 0) - cached_tokens - cache_write_tokens),
                output=usage.get("output_tokens") or 0,
                cache_read=cached_tokens,
                cache_write=cache_write_tokens,
                reasoning=output_tokens_details.get("reasoning_tokens") or 0,
                total_tokens=usage.get("total_tokens") or 0,
            )
        calculate_cost(model, output.usage)
        if options and options.apply_service_tier_pricing:
            if options.resolve_service_tier:
                service_tier = options.resolve_service_tier(
                    response.get("service_tier"), options.service_tier
                )
            else:
                service_tier = response.get("service_tier") or options.service_tier
            options.apply_service_tier_pricing(output.usage, service_tier)
        # 把 status 映射为 stop reason。对 incomplete 响应保留 provider 给出的具体原因，
        # 使 max-output 截断与内容过滤保持可区分。
        status = response.get("status")
        incomplete_details = response.get("incomplete_details") or {}
        incomplete_reason = (
            incomplete_details.get("reason")
            if isinstance(incomplete_details.get("reason"), str)
            else None
        )
        output.raw_stop_reason = f"{status}.{incomplete_reason}" if incomplete_reason else status
        mapped_stop = map_stop_reason(status, incomplete_reason)
        output.stop_reason = mapped_stop.stop_reason
        if mapped_stop.error_message is None:
            output.error_message = None
        else:
            output.error_message = mapped_stop.error_message
        if output.stop_reason == "stop" and any(
            getattr(block, "type", None) == "toolCall" for block in output.content
        ):
            output.stop_reason = "toolUse"

    async for event in openai_stream:
        if not isinstance(event, dict):
            continue
        if options and options.on_provider_stream_event:
            await maybe_await(options.on_provider_stream_event(event, model))
        event_type = event.get("type")
        if event_type == "response.created":
            output.response_id = event["response"]["id"]
        elif event_type == "response.output_item.added":
            create_slot(event.get("output_index"), event.get("item") or {})
        elif event_type == "response.reasoning_summary_text.delta":
            slot = get_slot(event.get("output_index"), "thinking")
            if slot is None:
                continue
            slot.block.thinking += event.get("delta") or ""
            stream.push(thinking_delta_event(slot.content_index, event.get("delta") or "", output))
        elif event_type == "response.reasoning_summary_part.done":
            slot = get_slot(event.get("output_index"), "thinking")
            if slot is None:
                continue
            slot.block.thinking += "\n\n"
            stream.push(thinking_delta_event(slot.content_index, "\n\n", output))
        elif event_type == "response.reasoning_text.delta":
            slot = get_slot(event.get("output_index"), "thinking")
            if slot is None:
                continue
            slot.block.thinking += event.get("delta") or ""
            stream.push(thinking_delta_event(slot.content_index, event.get("delta") or "", output))
        elif event_type == "response.output_text.delta":
            slot = get_slot(event.get("output_index"), "text")
            if slot is None:
                continue
            slot.block.text += event.get("delta") or ""
            stream.push(text_delta_event(slot.content_index, event.get("delta") or "", output))
        elif event_type == "response.refusal.delta":
            slot = get_slot(event.get("output_index"), "text")
            if slot is None:
                continue
            slot.block.text += event.get("delta") or ""
            stream.push(text_delta_event(slot.content_index, event.get("delta") or "", output))
        elif event_type == "response.function_call_arguments.delta":
            slot = get_slot(event.get("output_index"), "toolCall")
            if slot is None or getattr(slot.block, "partial_json", None) is None:
                continue
            slot.block.partial_json += event.get("delta") or ""
            slot.block.arguments = parse_streaming_json(slot.block.partial_json)
            push_tool_call_delta(slot, event.get("delta") or "")
        elif event_type == "response.function_call_arguments.done":
            slot = get_slot(event.get("output_index"), "toolCall")
            if slot is None or getattr(slot.block, "partial_json", None) is None:
                continue
            previous_partial_json = slot.block.partial_json
            arguments = event.get("arguments") or ""
            slot.block.partial_json = arguments
            slot.block.arguments = parse_streaming_json(slot.block.partial_json)
            if arguments.startswith(previous_partial_json):
                delta = arguments[len(previous_partial_json) :]
                if len(delta) > 0:
                    push_tool_call_delta(slot, delta)
        elif event_type == "response.custom_tool_call_input.delta":
            slot = get_slot(event.get("output_index"), "toolCall")
            if slot is None or not getattr(slot.block, "custom_input", None):
                continue
            push_tool_call_delta(
                slot,
                _append_custom_tool_call_input(
                    slot.block, _get_custom_tool_call_input(slot.block) + (event.get("delta") or ""), False
                ),
            )
        elif event_type == "response.custom_tool_call_input.done":
            slot = get_slot(event.get("output_index"), "toolCall")
            if slot is None or not getattr(slot.block, "custom_input", None):
                continue
            push_tool_call_delta(
                slot, _append_custom_tool_call_input(slot.block, event.get("input") or "", True)
            )
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            apply_message_phase_stop_reason(item)
            slot = get_or_create_slot(event.get("output_index"), item)
            item_type = item.get("type")
            if item_type == "reasoning" and slot is not None and slot.type == "thinking":
                summary_text = "\n\n".join(
                    entry.get("text", "") for entry in (item.get("summary") or [])
                ) or ""
                content_text = "\n\n".join(
                    entry.get("text", "") for entry in (item.get("content") or [])
                ) or ""
                slot.block.thinking = summary_text or content_text or slot.block.thinking
                slot.block.thinking_signature = _json_dumps(item)
                reasoning_blocks_by_id[item.get("id")] = slot.block
                stream.push(thinking_end_event(slot.content_index, slot.block.thinking, output))
                output_slots.pop(event.get("output_index"), None)
            elif item_type == "message" and slot is not None and slot.type == "text":
                slot.block.text = (
                    "".join(
                        (entry.get("text") if entry.get("type") == "output_text" else entry.get("refusal")) or ""
                        for entry in (item.get("content") or [])
                    )
                    or ""
                )
                slot.block.text_signature = encode_text_signature_v1(item.get("id"), item.get("phase"))
                stream.push(text_end_event(slot.content_index, slot.block.text, output))
                output_slots.pop(event.get("output_index"), None)
            elif (
                item_type == "function_call"
                and slot is not None
                and slot.type == "toolCall"
                and getattr(slot.block, "partial_json", None) is not None
            ):
                slot.block.arguments = parse_streaming_json(
                    item.get("arguments") or slot.block.partial_json or "{}"
                )
                if item.get("namespace") is not None:
                    slot.block.namespace = item["namespace"]
                # 就地定稿并删除临时缓冲区，使重放只携带解析后的参数。
                del slot.block.partial_json
                stream.push(tool_call_end_event(slot.content_index, slot.block, output))
                output_slots.pop(event.get("output_index"), None)
            elif (
                item_type == "custom_tool_call"
                and slot is not None
                and slot.type == "toolCall"
                and getattr(slot.block, "custom_input", None)
            ):
                push_tool_call_delta(
                    slot,
                    _append_custom_tool_call_input(
                        slot.block,
                        item.get("input") or _get_custom_tool_call_input(slot.block),
                        True,
                    ),
                )
                if item.get("namespace") is not None:
                    slot.block.namespace = item["namespace"]
                del slot.block.custom_input
                stream.push(tool_call_end_event(slot.content_index, slot.block, output))
                output_slots.pop(event.get("output_index"), None)
        elif event_type in ("response.completed", "response.incomplete"):
            finalize_response(event.get("response") or {})
        elif event_type == "error":
            raise RuntimeError(f"Error Code {event.get('code')}: {event.get('message')}")
        elif event_type == "response.failed":
            saw_terminal_response_event = True
            response = event.get("response") or {}
            output.raw_stop_reason = response.get("status")
            error = response.get("error")
            details = response.get("incomplete_details")
            if error:
                message = f"{error.get('code') or 'unknown'}: {error.get('message') or 'no message'}"
            elif details and details.get("reason"):
                message = f"incomplete: {details['reason']}"
            else:
                message = "Unknown error (no error details in response)"
            raise RuntimeError(message)

    if not saw_terminal_response_event:
        raise RuntimeError("OpenAI Responses stream ended before a terminal response event")
    # agent 会执行最终消息里的每一个工具调用。拒绝移交那些没有收到 output_item.done
    # 的调用：它们的参数可能被截断或错乱。已完成的调用其临时缓冲区已被移除。
    if output.stop_reason == "toolUse":
        for block in output.content:
            if getattr(block, "type", None) != "toolCall":
                continue
            if (
                getattr(block, "partial_json", None) is not None
                or getattr(block, "custom_input", None) is not None
            ):
                raise RuntimeError(
                    f"OpenAI Responses stream completed with an unfinished tool call: "
                    f"{block.name} ({block.id})"
                )


def map_stop_reason(status: str | None, incomplete_reason: str | None = None) -> _MappedStopReason:
    """把 Responses 的 ``status`` 映射为 pi 的 stop reason。

    Args:
        status: Responses 响应状态；为空时按正常结束处理。
        incomplete_reason: ``incomplete_details.reason``，用于区分截断与内容过滤。

    Returns:
        映射后的 :class:`_MappedStopReason`。

    Raises:
        RuntimeError: 遇到未知状态时。
    """
    if not status:
        return _MappedStopReason("stop")
    if status == "completed":
        return _MappedStopReason("stop")
    if status == "incomplete":
        if incomplete_reason == "max_output_tokens":
            return _MappedStopReason("length")
        return _MappedStopReason(
            "error",
            f"Response incomplete: {incomplete_reason}"
            if incomplete_reason
            else "Response incomplete without a provider reason",
        )
    if status in ("failed", "cancelled"):
        return _MappedStopReason("error")
    # 这两个状态比较诡异……
    if status in ("in_progress", "queued"):
        return _MappedStopReason("stop")
    raise RuntimeError(f"Unhandled stop reason: {status}")
