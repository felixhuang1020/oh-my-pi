"""provider adapter 共用的选项构建逻辑。

移植自 ``packages/ai/src/api/simple-options.ts``。各 ``stream_simple`` 实现负责把
provider 无关的 :class:`~pi_ai.types.SimpleStreamOptions` 转换为 provider 的
:class:`~pi_ai.types.StreamOptions`：把 ``max_tokens`` 钳制到 context window 的剩余
空间，并让 thinking budget 落在同一个 response ceiling 之内。
"""

from __future__ import annotations

from ..types import (
    Model,
    SimpleStreamOptions,
    StreamOptions,
    ThinkingBudgets,
    ThinkingLevel,
    TranscriptContext,
)
from ..utils.estimate import estimate_context_tokens

__all__ = [
    "CONTEXT_SAFETY_TOKENS",
    "DEFAULT_THINKING_BUDGETS",
    "MIN_ANSWER_TOKENS",
    "MIN_MAX_TOKENS",
    "adjust_max_tokens_for_thinking",
    "build_base_options",
    "clamp_max_tokens_to_context",
    "clamp_reasoning",
    "clamp_thinking_budget_to_answer_room",
    "thinking_budget_for_level",
]

#: 把 ``max_tokens`` 钳制到 context window 时始终保留的余量。
CONTEXT_SAFETY_TOKENS = 4096
#: 输出上限的最小值，避免钳制后得到 0 或负数。
MIN_MAX_TOKENS = 1

#: thinking budget 与回答共享同一个 response ceiling 时，为回答保留的最少 token。
MIN_ANSWER_TOKENS = 1024

#: 各 thinking 等级的默认 token budget。
DEFAULT_THINKING_BUDGETS: ThinkingBudgets = ThinkingBudgets(
    minimal=1024,
    low=2048,
    medium=8192,
    high=16384,
)


def clamp_max_tokens_to_context(model: Model, context: TranscriptContext, max_tokens: int) -> int:
    """缩小 ``max_tokens``，使请求能放进模型的 context window。

    Args:
        model: 请求将要发往的模型。
        context: 已构建的 transcript。
        max_tokens: 调用方期望的输出上限。

    Returns:
        不超过剩余空间、且不小于 :data:`MIN_MAX_TOKENS` 的输出上限。
    """
    if model.context_window <= 0:
        return max(MIN_MAX_TOKENS, max_tokens)
    available = model.context_window - estimate_context_tokens(context).tokens - CONTEXT_SAFETY_TOKENS
    return min(max_tokens, max(MIN_MAX_TOKENS, available))


def build_base_options(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None = None,
    api_key: str | None = None,
) -> StreamOptions:
    """把 simple options 投影为完整的 provider 选项集。

    Args:
        model: 请求将要发往的模型。
        context: 已构建的 transcript，用于估算上下文占用。
        options: provider 无关的简单选项；``None`` 时全部取默认值。
        api_key: 已解析好的 API key，优先于 ``options.api_key``。

    Returns:
        可直接交给 provider adapter 的 :class:`~pi_ai.types.StreamOptions`。
    """
    return StreamOptions(
        temperature=options.temperature if options else None,
        sampling_params=options.sampling_params if options else None,
        max_tokens=clamp_max_tokens_to_context(
            model, context, options.max_tokens if options and options.max_tokens is not None else model.max_tokens
        ),
        signal=options.signal if options else None,
        telemetry_context=options.telemetry_context if options else None,
        api_key=api_key or (options.api_key if options else None),
        fetch=options.fetch if options else None,
        transport=options.transport if options else None,
        cache_retention=options.cache_retention if options else None,
        session_id=options.session_id if options else None,
        headers=options.headers if options else None,
        on_payload=options.on_payload if options else None,
        on_response=options.on_response if options else None,
        on_provider_stream_event=options.on_provider_stream_event if options else None,
        timeout_ms=options.timeout_ms if options else None,
        websocket_connect_timeout_ms=options.websocket_connect_timeout_ms if options else None,
        max_retries=options.max_retries if options else None,
        max_retry_delay_ms=options.max_retry_delay_ms if options else None,
        metadata=options.metadata if options else None,
        env=options.env if options else None,
        extra=options.extra if options else {},
    )


def clamp_reasoning(effort: ThinkingLevel | str | None) -> str | None:
    """对按 token 预算的 provider，把扩展的 ``xhigh``/``max`` 等级折算为 ``high``。

    Args:
        effort: 调用方请求的 thinking 等级。

    Returns:
        折算后的等级；无需折算时原样返回（含 ``None``）。
    """
    if effort in ("xhigh", "max"):
        return "high"
    return effort


def thinking_budget_for_level(reasoning_level: str, custom_budgets: ThinkingBudgets | None = None) -> int:
    """``reasoning_level`` 对应的 token budget：默认值可被 ``custom_budgets`` 覆盖。

    Args:
        reasoning_level: 请求的 thinking 等级。
        custom_budgets: 可选的按等级覆盖值；为 ``None`` 的字段沿用默认值。

    Returns:
        该等级的 token budget；未知等级回退到 ``medium`` 的默认值。
    """
    budgets = {
        "minimal": DEFAULT_THINKING_BUDGETS.minimal,
        "low": DEFAULT_THINKING_BUDGETS.low,
        "medium": DEFAULT_THINKING_BUDGETS.medium,
        "high": DEFAULT_THINKING_BUDGETS.high,
    }
    if custom_budgets is not None:
        budgets.update(
            {
                level: value
                for level, value in (
                    ("minimal", custom_budgets.minimal),
                    ("low", custom_budgets.low),
                    ("medium", custom_budgets.medium),
                    ("high", custom_budgets.high),
                )
                if value is not None
            }
        )
    level = clamp_reasoning(reasoning_level)
    return budgets.get(level or "", DEFAULT_THINKING_BUDGETS.medium or 0)


def clamp_thinking_budget_to_answer_room(thinking_budget: int, ceiling: int) -> int:
    """限制 thinking budget，使其在 ``ceiling`` 之下至少留下 :data:`MIN_ANSWER_TOKENS`。

    Args:
        thinking_budget: 期望的 thinking token budget。
        ceiling: thinking 与回答共享的 response ceiling。

    Returns:
        钳制后的 thinking budget，最小为 0。
    """
    return min(thinking_budget, max(0, ceiling - MIN_ANSWER_TOKENS))


def adjust_max_tokens_for_thinking(
    base_max_tokens: int | None,
    model_max_tokens: int,
    reasoning_level: str,
    custom_budgets: ThinkingBudgets | None = None,
) -> tuple[int, int]:
    """在 thinking 与回答之间分配共享的 response ceiling。

    Args:
        base_max_tokens: 调用方显式指定的上限；``None`` 表示使用模型上限。
        model_max_tokens: 模型自身的上限。
        reasoning_level: 请求的 thinking 等级。
        custom_budgets: 可选的按等级覆盖值。

    Returns:
        ``(max_tokens, thinking_budget)``。
    """
    thinking_budget = thinking_budget_for_level(reasoning_level, custom_budgets)
    max_tokens = (
        model_max_tokens if base_max_tokens is None else min(base_max_tokens + thinking_budget, model_max_tokens)
    )
    if max_tokens <= thinking_budget:
        thinking_budget = clamp_thinking_budget_to_answer_room(thinking_budget, max_tokens)
    return max_tokens, thinking_budget
