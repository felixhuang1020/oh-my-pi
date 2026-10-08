"""上下文溢出（context overflow）检测.

移植自 ``packages/ai/src/utils/overflow.ts``。识别三种信号：错误信息匹配
provider 特定模式；响应成功但 input usage 已超过上下文窗口（静默溢出）；
stop reason 为 ``length``、输出为零且输入已填满窗口（服务端截断）。
"""

from __future__ import annotations

import re

from ..types import AssistantMessage

__all__ = [
    "NON_OVERFLOW_PATTERNS",
    "OVERFLOW_PATTERNS",
    "get_overflow_patterns",
    "is_context_overflow",
    "is_recoverable_length",
]

#: 输入超出模型上下文窗口时返回的消息。
OVERFLOW_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"prompt (?:is )?too long",  # Anthropic 与 z.ai 的 token 溢出
        r"prompt exceeds max length",  # z.ai 中国区端点的 token 溢出
        r"request_too_large",  # Anthropic 请求字节数溢出（HTTP 413）
        r"input is too long for requested model",  # Amazon Bedrock 输入过长
        r"exceeds the context window",  # OpenAI（Completions 与 Responses API）
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
        r"input token count.*exceeds the maximum",  # Google（Gemini）token 超限
        r"maximum prompt length is \d+",  # xAI（Grok）prompt 长度上限
        r"reduce the length of the messages",  # Groq 建议缩短消息
        r"maximum context length is \d+ tokens",  # OpenRouter（多数后端）
        # OpenRouter/Poolside 的最大允许输入长度
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
        # Together AI 的输入与上下文长度比较
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        r"exceeds the limit of \d+",  # GitHub Copilot 的限额
        r"exceeds the available context size",  # llama.cpp server 可用上下文不足
        r"greater than the context length",  # LM Studio 的上下文长度
        r"context window exceeds limit",  # MiniMax 上下文窗口超限
        r"exceeded model token limit",  # Kimi For Coding 的模型 token 上限
        r"too large for model with \d+ maximum context length",  # Mistral 的上下文长度过小
        # DS4 server：prompt 与配置的上下文大小不符
        r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",
        r"model_context_window_exceeded",  # z.ai 非标准 finish_reason，以错误文本出现
        r"prompt too long; exceeded (?:max )?context length",  # Ollama 显式溢出错误
        r"range of input length should be",  # DashScope / Qwen Token Plan 输入长度范围
        r"context[_ ]length[_ ]exceeded",  # 通用兜底
        r"too many tokens",  # 通用兜底
        r"token limit exceeded",  # 通用兜底
    )
)

#: 本会匹配溢出模式、但实际并非溢出的错误。
NON_OVERFLOW_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(?:Throttling error|Service unavailable):",  # AWS Bedrock 的非溢出错误
        r"rate limit",  # 通用限流
        r"too many requests",  # 通用 HTTP 429 风格
    )
)


def is_context_overflow(message: AssistantMessage, context_window: int | None = None) -> bool:
    """判断 ``message`` 是否表示请求超出了模型的上下文窗口.

    Args:
        message: 待检查的 assistant message。
        context_window: 可选的窗口大小；给出后可检测静默溢出（z.ai）
            和 length 停止型溢出（Xiaomi MiMo）。

    Returns:
        判定为上下文溢出时返回 ``True``。
    """
    error_message = message.error_message
    if message.stop_reason == "error" and error_message:
        # 跳过匹配已知非溢出模式（限流/节流）的消息。
        if not any(pattern.search(error_message) for pattern in NON_OVERFLOW_PATTERNS):
            if any(pattern.search(error_message) for pattern in OVERFLOW_PATTERNS):
                return True

    usage = message.usage
    if context_window and message.stop_reason == "stop":
        if usage.input + usage.cache_read > context_window:
            return True

    if context_window and message.stop_reason == "length" and usage.output == 0:
        if usage.input + usage.cache_read >= context_window * 0.99:
            return True

    return False


def is_recoverable_length(message: AssistantMessage, desired_max_output: int) -> bool:
    """判断 length 停止是否发生在预期输出上限之下.

    这类响应可能由上下文压力或 provider 侧截断造成，调用方可据此做一次
    有界的压缩重试。``desired_max_output`` 必须是任何基于上下文的钳制
    之前的原始上限。

    Args:
        message: 待检查的 assistant message。
        desired_max_output: 调用方期望的输出上限（未经过上下文钳制）。

    Returns:
        ``length`` 停止且输出未达预期上限时返回 ``True``。
    """
    return (
        message.stop_reason == "length"
        and desired_max_output > 0
        and message.usage.output < desired_max_output
    )


def get_overflow_patterns() -> list[re.Pattern[str]]:
    """返回溢出模式列表，供测试使用.

    Returns:
        :data:`OVERFLOW_PATTERNS` 的可变副本。
    """
    return list(OVERFLOW_PATTERNS)
