"""Google Generative AI provider 的共享工具。

移植自 ``packages/ai/src/api/google-shared.ts``。

TypeScript 原版构建在 ``@google/genai`` SDK 之上。本移植没有官方 Google SDK，因此
该 SDK 承担的两项职责在 :mod:`httpx` 之上复刻：

* :func:`to_generate_content_body` 完成 SDK 在发请求前做的
  ``GenerateContentConfig`` → REST body 映射。
* :func:`open_google_generate_content_stream` 完成 SDK 的
  ``models.generateContentStream`` 调用（``:streamGenerateContent?alt=sse``），
  并产出解码后的 ``GenerateContentResponse`` 块。

其余均为直接移植，包括 SDK 的 ``FinishReason``、``FunctionCallingConfigMode`` 和
``ThinkingLevel`` 枚举的 API 级字符串值（SDK 将这些枚举序列化为其名称，而线上协议
使用的正是这些名称）。
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

import httpx

from ..models import clamp_thinking_level
from ..types import (
    Model,
    StopReason,
    StreamOptions,
    ThinkingLevel,
    Tool,
    TranscriptContext,
)
from ..utils.abort import AbortSignal, operation_signal
from ..utils.http import create_client, read_error_body
from ..utils.provider_retry import ProviderRetryOptions, retry_provider_request
from ..utils.sanitize_unicode import sanitize_surrogates
from ..utils.sse import iter_sse_events
from ..utils.transcript import collapse_system_messages, without_initial_system_message
from .constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from .transform_messages import transform_messages

__all__ = [
    "GOOGLE_GENERATIVE_AI_API",
    "GoogleApiError",
    "GoogleApiThinkingLevel",
    "GoogleContentStream",
    "GoogleThinkingOptions",
    "ResolvedGoogleThinkingLevel",
    "convert_messages",
    "convert_tools",
    "get_disabled_google_thinking_config",
    "is_thinking_part",
    "map_stop_reason",
    "map_stop_reason_string",
    "map_tool_choice",
    "open_google_generate_content_stream",
    "requires_tool_call_id",
    "resolve_google_function_calling_mode",
    "resolve_google_thinking_level",
    "retain_thought_signature",
    "retry_google_request",
    "supports_google_strict_tool_sampling",
    "to_generate_content_body",
    "to_google_sdk_thinking_level",
    "to_google_thinking_level",
    "uses_google_thinking_level",
]

T = TypeVar("T")

#: 本共享模块服务的聊天 API id。
GOOGLE_GENERATIVE_AI_API = "google-generative-ai"


class GoogleApiThinkingLevel(StrEnum):
    """Gemini 3 模型的 thinking 等级。

    镜像 Google 的 ``ThinkingLevel`` 枚举值，SDK 会原样序列化。
    """

    THINKING_LEVEL_UNSPECIFIED = "THINKING_LEVEL_UNSPECIFIED"
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ResolvedGoogleThinkingLevel(StrEnum):
    """Google 接受的 pi thinking 等级（``Exclude<ThinkingLevel, "xhigh" | "max">``）。"""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


#: SDK 的 ``ThinkingLevel`` 枚举值，以 pi 侧的 Google 等级为键。
_GOOGLE_SDK_THINKING_LEVEL_MAP: dict[GoogleApiThinkingLevel, str] = {
    GoogleApiThinkingLevel.THINKING_LEVEL_UNSPECIFIED: "THINKING_LEVEL_UNSPECIFIED",
    GoogleApiThinkingLevel.MINIMAL: "MINIMAL",
    GoogleApiThinkingLevel.LOW: "LOW",
    GoogleApiThinkingLevel.MEDIUM: "MEDIUM",
    GoogleApiThinkingLevel.HIGH: "HIGH",
}


@dataclass
class GoogleThinkingOptions:
    """两个 Google 适配器共享的按请求 thinking 控制。

    Attributes:
        enabled: 是否启用 thinking。
        budget_tokens: token 预算；``-1`` 表示动态，``0`` 表示关闭。
        level: Gemini 3 使用的离散 thinking 等级。
    """

    enabled: bool = False
    #: ``-1`` 表示动态，``0`` 表示关闭。
    budget_tokens: int | None = None
    level: GoogleApiThinkingLevel | str | None = None


def resolve_google_thinking_level(
    model: Model,
    level: ThinkingLevel | str,
) -> ResolvedGoogleThinkingLevel:
    """把受支持的 pi 等级或模型特定映射解析为标准 Google 等级。

    Args:
        model: 目标模型。
        level: pi 侧 thinking 等级。

    Returns:
        标准化的 :class:`ResolvedGoogleThinkingLevel`。

    Raises:
        ValueError: 当等级不在受支持范围内、且模型映射也无法解析出合法等级时。
    """
    mapping = model.thinking_level_map
    present = mapping is not None and level in mapping
    mapped = mapping.get(level) if mapping is not None else None
    resolved_level = mapped.lower() if isinstance(mapped, str) else level
    if resolved_level in ("minimal", "low", "medium", "high"):
        return ResolvedGoogleThinkingLevel(resolved_level)
    # 对应 `String(mapped)`：键不存在渲染为 "undefined"，显式 JSON null 渲染为 "null"，
    # 其他值原样渲染（因此 `"extreme"` 这类非法映射会被逐字报告）。
    if not present:
        rendered = "undefined"
    else:
        rendered = "null" if mapped is None else str(mapped)
    raise ValueError(
        f"Unsupported Google thinking level mapping for {model.provider}/{model.id}: "
        f"{level} -> {rendered}"
    )


def uses_google_thinking_level(model: Model) -> bool:
    """该模型是否使用 Gemini 的离散 ``thinkingLevel`` 控制。

    支持的等级来自模型的 ``thinkingLevelMap``；此函数只决定 Google 线上格式。

    Args:
        model: 目标模型。

    Returns:
        使用离散 ``thinkingLevel`` 时返回 ``True``，否则使用 token 预算控制。
    """
    model_id = model.id.lower()
    return (
        # 匹配带或不带次版本号的 Gemini 3 Pro/Flash ID，例如
        # gemini-3-flash-preview、gemini-3.1-pro-preview 和 gemini-3.8-flash。
        re.search(r"gemini-3(?:\.\d+)?-(?:pro|flash)", model_id) is not None
        or model_id == "gemini-flash-latest"
        or model_id == "gemini-flash-lite-latest"
        # 匹配两种托管 Gemma 4 命名形式：gemma-4-* 与 gemma4-*。
        or re.search(r"gemma-?4", model_id) is not None
    )


def to_google_thinking_level(level: ResolvedGoogleThinkingLevel) -> GoogleApiThinkingLevel:
    """把解析后的 pi 等级映射到 Google API 级枚举。

    Args:
        level: 已解析的 Google thinking 等级。

    Returns:
        对应的 :class:`GoogleApiThinkingLevel`。
    """
    if level == ResolvedGoogleThinkingLevel.MINIMAL:
        return GoogleApiThinkingLevel.MINIMAL
    if level == ResolvedGoogleThinkingLevel.LOW:
        return GoogleApiThinkingLevel.LOW
    if level == ResolvedGoogleThinkingLevel.MEDIUM:
        return GoogleApiThinkingLevel.MEDIUM
    return GoogleApiThinkingLevel.HIGH


def to_google_sdk_thinking_level(level: GoogleApiThinkingLevel) -> str:
    """把 API 级枚举映射到 SDK ``ThinkingLevel`` 序列化后的值。

    Args:
        level: API 级枚举值。

    Returns:
        SDK 会原样序列化到线上协议的字符串。
    """
    return _GOOGLE_SDK_THINKING_LEVEL_MAP[level]


def get_disabled_google_thinking_config(model: Model) -> dict[str, Any]:
    """为 ``model`` 关闭推理所需的 ``thinkingConfig``。

    Args:
        model: 目标模型。

    Returns:
        可直接放入 ``generationConfig.thinkingConfig`` 的字典。
    """
    if not uses_google_thinking_level(model):
        return {"thinkingBudget": 0}

    fallback = clamp_thinking_level(model, "off")
    if fallback == "off":
        return {"thinkingBudget": 0}

    resolved_level = resolve_google_thinking_level(model, fallback)
    api_level = to_google_thinking_level(resolved_level)
    return {"thinkingLevel": to_google_sdk_thinking_level(api_level)}


def is_thinking_part(part: dict[str, Any]) -> bool:
    """流式 Gemini ``Part`` 是否应视为 "thinking"。

    ``thought: true`` 是 thinking 内容（思考摘要）的权威标记。
    ``thoughtSignature`` 可出现在任意 part 类型上，本身*不*表示该 part 是 thinking 内容。

    参见: https://ai.google.dev/gemini-api/docs/thought-signatures

    Args:
        part: 单个流式响应 ``Part``。

    Returns:
        该 part 是 thinking 内容时返回 ``True``。
    """
    return part.get("thought") is True


def retain_thought_signature(existing: str | None, incoming: str | None) -> str | None:
    """为当前流式块保留最后一个非空 thought signature。

    某些后端只在给定块的第一个 delta 上发送 ``thoughtSignature``，后续 delta 可能省略。
    这不会在不同响应 part 之间合并或迁移签名。

    Args:
        existing: 已保留的签名。
        incoming: 本次 delta 携带的签名。

    Returns:
        非空的 ``incoming``；否则返回 ``existing``。
    """
    if isinstance(incoming, str) and len(incoming) > 0:
        return incoming
    return existing


# Google API 要求 thought signature 为 base64（TYPE_BYTES）。
_BASE64_SIGNATURE_PATTERN = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _is_valid_thought_signature(signature: str | None) -> bool:
    """判断签名是否为长度对齐的合法 base64。

    Args:
        signature: 待校验的签名。

    Returns:
        长度是 4 的倍数且字符集合法时返回 ``True``。
    """
    if not signature:
        return False
    if len(signature) % 4 != 0:
        return False
    return _BASE64_SIGNATURE_PATTERN.fullmatch(signature) is not None


def _resolve_thought_signature(is_same_provider_and_model: bool, signature: str | None) -> str | None:
    """仅保留同 provider/模型且 base64 合法的签名。

    Args:
        is_same_provider_and_model: 消息是否来自相同 provider 与模型。
        signature: 待处理的签名。

    Returns:
        合法的签名；否则返回 ``None``。
    """
    return signature if is_same_provider_and_model and _is_valid_thought_signature(signature) else None


def requires_tool_call_id(model_id: str) -> bool:
    """经由 Google API 调用、要求 function call/response 显式携带 tool call ID 的模型。

    Args:
        model_id: 模型 ID。

    Returns:
        需要显式携带 tool call ID 时返回 ``True``。
    """
    gemini_major_version = _get_gemini_major_version(model_id)
    return (
        model_id.startswith("claude-")
        or model_id.startswith("gpt-oss-")
        or (gemini_major_version is not None and gemini_major_version >= 3)
    )


def _get_gemini_major_version(model_id: str) -> int | None:
    """解析 Gemini 模型 ID 中的主版本号。

    Args:
        model_id: 模型 ID。

    Returns:
        主版本号；不是 Gemini 模型时为 ``None``。
    """
    match = re.match(r"^gemini(?:-live)?-(\d+)", model_id.lower())
    if match is None:
        return None
    return int(match.group(1))


def convert_messages(model: Model, context: TranscriptContext) -> list[dict[str, Any]]:
    """把内部消息转换为 Gemini ``Content[]`` 格式。

    Args:
        model: 目标模型。
        context: 会话上下文。

    Returns:
        可直接放入请求体 ``contents`` 的消息列表。
    """
    # Gemini 没有对话中途的 system 消息；开头的 prompt 作为 systemInstruction 发送。
    conversation = without_initial_system_message(collapse_system_messages(context).messages)
    contents: list[dict[str, Any]] = []

    def normalize_tool_call_id(tool_call_id: str, _model: Model, _message: Any) -> str:
        """按需把 tool call ID 规范化为 Google 接受的字符集。

        Args:
            tool_call_id: 原始 tool call ID。
            _model: 忽略的模型参数，签名由 :func:`transform_messages` 约定。
            _message: 忽略的消息参数，签名由 :func:`transform_messages` 约定。

        Returns:
            规范化后的 tool call ID（最长 64 字符）。
        """
        if not requires_tool_call_id(model.id):
            return tool_call_id
        return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_call_id)[:64]

    transformed_messages = transform_messages(conversation, model, normalize_tool_call_id)

    for msg in transformed_messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                contents.append({"role": "user", "parts": [{"text": sanitize_surrogates(msg.content)}]})
            else:
                parts: list[dict[str, Any]] = [
                    {"text": sanitize_surrogates(item.text)}
                    for item in msg.content
                    if getattr(item, "type", None) == "text"
                ]
                if len(parts) == 0:
                    continue
                contents.append({"role": "user", "parts": parts})
        elif msg.role == "assistant":
            parts = []
            # 检查消息是否来自相同 provider 与模型——只有此时才保留 thinking 块。
            is_same_provider_and_model = msg.provider == model.provider and msg.model == model.id

            for block in msg.content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    thought_signature = _resolve_thought_signature(is_same_provider_and_model, block.text_signature)
                    # 跳过空文本块——除非它们带有 thought signature。Gemini 可能把签名附加在
                    # 可见文本为空的 part 上并要求原样回传；丢弃它会破坏推理链，模型会间歇性
                    # 地在任务中途以仅含思考的 STOP 结束回合。
                    if (not block.text or block.text.strip() == "") and not thought_signature:
                        continue
                    part: dict[str, Any] = {"text": sanitize_surrogates(block.text)}
                    if thought_signature:
                        part["thoughtSignature"] = thought_signature
                    parts.append(part)
                elif block_type == "thinking":
                    # 仅当 provider 与模型都相同时才保留为 thinking 块，
                    # 否则转为纯文本（不加标签，避免模型模仿标签）。
                    if is_same_provider_and_model:
                        thought_signature = _resolve_thought_signature(
                            is_same_provider_and_model, block.thinking_signature
                        )
                        # 与文本块同理：空 thinking 块仅在不带签名时才丢弃。
                        if (not block.thinking or block.thinking.strip() == "") and not thought_signature:
                            continue
                        part = {"thought": True, "text": sanitize_surrogates(block.thinking)}
                        if thought_signature:
                            part["thoughtSignature"] = thought_signature
                        parts.append(part)
                    else:
                        # 跨 provider/模型：签名不可用，空块照常丢弃。
                        if not block.thinking or block.thinking.strip() == "":
                            continue
                        parts.append({"text": sanitize_surrogates(block.thinking)})
                elif block_type == "toolCall":
                    thought_signature = _resolve_thought_signature(
                        is_same_provider_and_model, block.thought_signature
                    )
                    part = {
                        "functionCall": {
                            "name": block.name,
                            "args": block.arguments if block.arguments is not None else {},
                        }
                    }
                    if requires_tool_call_id(model.id):
                        part["functionCall"]["id"] = block.id
                    if thought_signature:
                        part["thoughtSignature"] = thought_signature
                    parts.append(part)

            if len(parts) == 0:
                continue
            contents.append({"role": "model", "parts": parts})
        elif msg.role == "toolResult":
            text_content = [c for c in msg.content if getattr(c, "type", None) == "text"]
            text_result = "\n".join(c.text for c in text_content)

            # 成功用 "output" 键、出错用 "error" 键，遵循 SDK 文档。
            response_value = sanitize_surrogates(text_result)

            include_id = requires_tool_call_id(model.id)
            function_response: dict[str, Any] = {
                "name": msg.tool_name,
                "response": {"error": response_value} if msg.is_error else {"output": response_value},
            }
            if include_id:
                function_response["id"] = msg.tool_call_id
            function_response_part: dict[str, Any] = {"functionResponse": function_response}

            # Cloud Code Assist API 要求所有 function response 位于同一个 user 轮次中。
            # 检查最后一条内容是否已是带 function response 的 user 轮次，是则合并。
            last_content = contents[-1] if contents else None
            if (
                last_content is not None
                and last_content.get("role") == "user"
                and any(part.get("functionResponse") for part in last_content.get("parts") or [])
            ):
                last_content["parts"].append(function_response_part)
            else:
                contents.append({"role": "user", "parts": [function_response_part]})

    return contents


_JSON_SCHEMA_META_DECLARATIONS = frozenset(
    [
        "$schema",
        "$id",
        "$anchor",
        "$dynamicAnchor",
        "$vocabulary",
        "$comment",
        "$defs",
        "definitions",  # draft-2019-09 之前的 $defs 等价物
    ]
)


def _sanitize_for_open_api(schema: Any) -> Any:
    """剥离 schema 中的元声明。

    Args:
        schema: 待处理的 JSON Schema 片段。

    Returns:
        移除 ``$schema``、``$id`` 等元声明后的结构；非字典原样返回。
    """
    if not isinstance(schema, dict):
        return schema

    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _JSON_SCHEMA_META_DECLARATIONS:
            continue
        result[key] = _sanitize_for_open_api(value)
    return result


def convert_tools(
    tools: Sequence[Tool],
    use_parameters: bool = False,
    supports_strict_mode: bool = True,
) -> list[dict[str, Any]] | None:
    """把工具转换为 Gemini function declarations 格式。

    默认使用 ``parametersJsonSchema``，支持完整 JSON Schema（含 anyOf、oneOf、const 等）。
    将 ``use_parameters`` 设为 ``True`` 可改用旧版 ``parameters`` 字段（OpenAPI 3.03
    Schema）。Cloud Code Assist 上的 Claude 模型需要这样做：该 API 会把 ``parameters``
    转译为 Anthropic 的 ``input_schema``。

    Args:
        tools: 待转换的工具列表。
        use_parameters: 为 ``True`` 时使用旧版 ``parameters`` 字段。
        supports_strict_mode: 模型是否支持严格模式采样。

    Returns:
        ``tools`` 数组；没有工具时返回 ``None``。
    """
    if len(tools) == 0:
        return None
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        parameters = get_json_schema_tool_parameters(tool, strict)
        declaration: dict[str, Any] = {"name": tool.name, "description": tool.description}
        if use_parameters:
            declaration["parameters"] = _sanitize_for_open_api(parameters)
        else:
            declaration["parametersJsonSchema"] = parameters
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}]


def supports_google_strict_tool_sampling(model_id: str) -> bool:
    """Gemini 3+ 在受校验的 tool-calling 模式下强制要求函数必填参数。

    Args:
        model_id: 模型 ID。

    Returns:
        Gemini 3+ 返回 ``True``。
    """
    major_version = _get_gemini_major_version(model_id)
    return major_version is not None and major_version >= 3


def map_tool_choice(choice: str) -> str:
    """把 tool choice 字符串映射到 Gemini ``FunctionCallingConfigMode``。

    Args:
        choice: tool choice 字符串。

    Returns:
        ``AUTO``、``NONE`` 或 ``ANY``；未知值回退为 ``AUTO``。
    """
    if choice == "auto":
        return "AUTO"
    if choice == "none":
        return "NONE"
    if choice == "any":
        return "ANY"
    return "AUTO"


def resolve_google_function_calling_mode(
    tools: Sequence[Tool],
    tool_choice: str | None,
    supports_strict_mode: bool,
) -> str | None:
    """应发送的 ``functionCallingConfig.mode``；为 ``None`` 则不设置该字段。

    Args:
        tools: 当前可用工具列表。
        tool_choice: 调用方指定的 tool choice。
        supports_strict_mode: 模型是否支持严格模式采样。

    Returns:
        模式字符串；无需设置该字段时返回 ``None``。
    """
    use_strict_mode = any(resolve_json_schema_strict_sampling(tool, supports_strict_mode) is True for tool in tools)
    if tool_choice == "none" or tool_choice == "any":
        return map_tool_choice(tool_choice)
    if use_strict_mode:
        return "VALIDATED"
    return map_tool_choice(tool_choice) if tool_choice else None


def map_stop_reason(reason: str) -> StopReason:
    """把 Gemini ``FinishReason`` 映射到本项目的 :class:`StopReason`。

    Args:
        reason: Gemini 返回的 ``FinishReason``。

    Returns:
        对应的 :class:`StopReason`；安全与内容类原因统一映射为 ``ERROR``。

    Raises:
        ValueError: 遇到未覆盖的 finish reason 时。
    """
    if reason == "STOP":
        return StopReason.STOP
    if reason == "MAX_TOKENS":
        return StopReason.LENGTH
    if reason in (
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "SAFETY",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
        "RECITATION",
        "FINISH_REASON_UNSPECIFIED",
        "OTHER",
        "LANGUAGE",
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "TOO_MANY_TOOL_CALLS",
        "NO_IMAGE",
    ):
        return StopReason.ERROR
    raise ValueError(f"Unhandled stop reason: {reason}")


def map_stop_reason_string(reason: str) -> StopReason:
    """把字符串形式的 finish reason 映射到 :class:`StopReason`（用于原始 API 响应）。

    Args:
        reason: 原始 finish reason 字符串。

    Returns:
        ``STOP``/``MAX_TOKENS`` 之外的取值统一映射为 ``ERROR``。
    """
    if reason == "STOP":
        return StopReason.STOP
    if reason == "MAX_TOKENS":
        return StopReason.LENGTH
    return StopReason.ERROR


async def retry_google_request(
    request: Callable[[], Awaitable[T]],
    options: StreamOptions | None = None,
) -> T:
    """按共享 provider 重试策略执行一次 Google 请求。

    对 408/409/429/5xx 按退避重试并遵循 retry-after，与 Anthropic 和 OpenAI 适配器用
    :func:`~pi_ai.utils.provider_retry.retry_provider_request` 包裹初始请求的方式一致。
    TypeScript 原版会规范化带 ``status`` 但没有 ``headers`` 属性的 SDK 错误；移植后的
    重试辅助只需 status，因此保留该规范化以保持一致，并显式附加 ``None`` 的 headers。

    Args:
        request: 发起请求并在成功时返回结果的可调用对象。
        options: 提供重试次数、退避与取消信号的 stream 选项。

    Returns:
        ``request`` 的返回值。

    Raises:
        Exception: 请求最终失败或等待期间被取消时，原样重新抛出。
    """

    async def normalized_request() -> T:
        """按 TypeScript 语义补齐缺失的 ``headers`` 属性后执行请求。"""
        try:
            return await request()
        except Exception as error:  # noqa: BLE001 - 规范化后重新抛出
            if hasattr(error, "status") and not hasattr(error, "headers"):
                error.headers = None  # type: ignore[attr-defined]  # 补齐上游规范化的 headers 属性。
            raise

    return await retry_provider_request(
        normalized_request,
        ProviderRetryOptions(
            max_retries=options.max_retries if options is not None else None,
            max_retry_delay_ms=options.max_retry_delay_ms if options is not None else None,
            signal=options.signal if options is not None else None,
        ),
    )


# --------------------------------------------------------------------------------------
# SDK 替代实现：请求序列化与流式响应
# --------------------------------------------------------------------------------------


class GoogleApiError(Exception):
    """非 2xx 的 Google API 响应。

    携带 ``status``/``status_code`` 与 ``headers``，供
    :func:`~pi_ai.utils.provider_retry.retry_provider_request` 分类，另有响应 ``body``
    用于错误报告。
    """

    def __init__(self, status: int, body: str, status_text: str = "", headers: Any = None) -> None:
        """记录 HTTP 状态、响应体与响应头。

        Args:
            status: HTTP 状态码。
            body: 错误响应体文本。
            status_text: HTTP reason phrase。
            headers: 原始响应头。
        """
        super().__init__(status_text or f"Request failed with status {status}")
        self.status = status
        self.status_code = status
        self.body = body
        self.headers = headers


def to_generate_content_body(params: dict[str, Any]) -> dict[str, Any]:
    """把 ``GenerateContentParameters`` 序列化为 REST 请求体。

    ``@google/genai`` SDK 原本在内部完成此映射；本移植在此处实现，使 ``onPayload``
    仍能看到与 TypeScript 中相同的 ``{model, contents, config}`` 形态。

    Args:
        params: ``{model, contents, config}`` 形态的请求参数。

    Returns:
        可直接作为 HTTP JSON body 发送的字典。
    """
    config = params.get("config") or {}
    body: dict[str, Any] = {"contents": params.get("contents") or []}

    generation_config: dict[str, Any] = {}
    if config.get("temperature") is not None:
        generation_config["temperature"] = config["temperature"]
    if config.get("maxOutputTokens") is not None:
        generation_config["maxOutputTokens"] = config["maxOutputTokens"]
    if config.get("thinkingConfig") is not None:
        generation_config["thinkingConfig"] = config["thinkingConfig"]
    if generation_config:
        body["generationConfig"] = generation_config

    system_instruction = config.get("systemInstruction")
    if system_instruction:
        if isinstance(system_instruction, str):
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        else:
            body["systemInstruction"] = system_instruction
    if config.get("tools"):
        body["tools"] = config["tools"]
    if config.get("toolConfig"):
        body["toolConfig"] = config["toolConfig"]
    return body


async def _await_or_abort(operation: Awaitable[T], signal: AbortSignal) -> T:
    """等待 ``operation``，在 ``signal`` 中止时取消它。

    :func:`~pi_ai.utils.abort.race_with_abort_signal` 刻意让被放弃的 awaitable 继续运行，
    以便其异常能被观察到。对响应体的读取则不同：若不取消，底层异步生成器会在 stream
    生命周期结束后继续运行，因此这里让 body 迭代取消进行中的读取。

    Args:
        operation: 待等待的 awaitable。
        signal: 中止信号。

    Returns:
        ``operation`` 的结果。

    Raises:
        BaseException: ``signal`` 中止时抛出其对应的中止异常。
    """
    task: asyncio.Future[T] = asyncio.ensure_future(operation)
    waiter = asyncio.ensure_future(signal.wait())
    try:
        done, _pending = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        task.cancel()
        waiter.cancel()
        raise
    if task in done:
        waiter.cancel()
        return task.result()
    task.cancel()
    try:
        await task
    except BaseException:  # noqa: BLE001 - 以下的中止原因才是权威
        pass
    raise signal.exception()


async def _iter_abortable_lines(response: httpx.Response, signal: AbortSignal) -> AsyncIterator[str]:
    """逐行产出响应体，一旦 ``signal`` 中止便停止。

    Args:
        response: 待读取的流式响应。
        signal: 中止信号。

    Yields:
        响应体中的每一行（不含行尾换行符）。
    """
    iterator = response.aiter_lines()
    while True:
        try:
            line = await _await_or_abort(anext(iterator), signal)
        except StopAsyncIteration:
            return
        yield line


class GoogleContentStream:
    """已解码为响应块的 ``streamGenerateContent`` SSE 实时响应。

    迭代产出的对象与 SDK ``generateContentStream`` 产出的 ``GenerateContentResponse``
    一致。用完后调用 :meth:`close`（各适配器均在 ``finally`` 块中调用）。

    Attributes:
        _client: 承载请求的 ``httpx.AsyncClient``。
        _response: 已建立但尚未读完的流式响应。
        _signal: 用于中止读取的中止信号。
    """

    __slots__ = ("_client", "_response", "_signal")

    def __init__(self, client: httpx.AsyncClient, response: httpx.Response, signal: AbortSignal) -> None:
        """保存客户端、流式响应与中止信号。

        Args:
            client: 承载请求的 ``httpx.AsyncClient``。
            response: 已建立但尚未读完的流式响应。
            signal: 用于中止读取的中止信号。
        """
        self._client = client
        self._response = response
        self._signal = signal

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        """返回解码后响应块的异步迭代器。

        Returns:
            该 stream 的异步迭代器。
        """
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[dict[str, Any]]:
        """解析 SSE 事件并逐块产出 ``GenerateContentResponse``。

        Yields:
            解码后的响应块；非 JSON 对象包装为 ``{"raw": ...}``。
        """
        async for event in iter_sse_events(_iter_abortable_lines(self._response, self._signal)):
            data = event.data.strip()
            if not data:
                continue
            if data == "[DONE]":
                return
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                yield parsed
            else:
                yield {"raw": parsed}

    async def close(self) -> None:
        """释放该 stream 背后的 HTTP 响应与客户端。"""
        try:
            await self._response.aclose()
        finally:
            await self._client.aclose()


async def open_google_generate_content_stream(
    *,
    url: str,
    headers: dict[str, str],
    params: dict[str, Any],
    options: StreamOptions | None = None,
) -> GoogleContentStream:
    """发起 ``models.generateContentStream`` 并返回解码后的块流。

    请求本身（连接 + 状态检查）在此发生，以便调用方用 :func:`retry_google_request`
    包裹，正如 TypeScript 适配器包裹 ``client.models.generateContentStream(params)`` 那样。

    Args:
        url: ``:streamGenerateContent`` 完整 URL。
        headers: 请求头。
        params: ``{model, contents, config}`` 形态的请求参数。
        options: 提供超时、fetch 与取消信号的 stream 选项。

    Returns:
        已建立连接的 :class:`GoogleContentStream`。

    Raises:
        GoogleApiError: 响应状态码不小于 400 时。
    """
    operation = operation_signal(options.signal if options is not None else None)
    signal = operation
    if options is not None and options.timeout_ms is not None:
        signal = AbortSignal.any([operation, AbortSignal.timeout(options.timeout_ms)])

    client = create_client(
        headers=headers,
        timeout_ms=options.timeout_ms if options is not None else None,
        fetch=options.fetch if options is not None else None,
    )
    try:
        request = client.build_request("POST", url, json=to_generate_content_body(params))
        response = await _await_or_abort(client.send(request, stream=True), signal)
    except BaseException:
        await client.aclose()
        raise

    if response.status_code >= 400:
        status = response.status_code
        status_text = response.reason_phrase
        response_headers = dict(response.headers)
        body_text = await read_error_body(response)
        await response.aclose()
        await client.aclose()
        raise GoogleApiError(status, body_text, status_text, response_headers)

    return GoogleContentStream(client, response, signal)
