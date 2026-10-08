"""移植 ``packages/ai/test/context-estimate.test.ts``。

验证 :func:`pi_ai.utils.estimate.estimate_context_tokens` 会忽略被后来插入的消息
覆盖的 assistant usage 块，并在插入内容已有响应后重新以它为锚点。
第二处断言通过 :func:`pi_ai.api.simple_options.build_base_options` 检查钳制后的
``max_tokens``。
"""

from __future__ import annotations

from pi_ai.api.simple_options import build_base_options
from pi_ai.types import AssistantMessage, Context, Model, ModelCost, StopReason, TextContent, Usage, UserMessage
from pi_ai.utils.estimate import estimate_context_tokens
from pi_ai.utils.transcript import normalize_context

MODEL = Model(
    id="test-model",
    name="Test Model",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=False,
    input=["text"],
    cost=ModelCost(),
    context_window=10_000,
    max_tokens=8_000,
)


def create_usage(total_tokens: int) -> Usage:
    return Usage(input=total_tokens, output=0, cache_read=0, cache_write=0, total_tokens=total_tokens)


def create_assistant(timestamp: int, total_tokens: int) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(text="kept")],
        api="openai-responses",
        provider="openai",
        model="test-model",
        usage=create_usage(total_tokens),
        stop_reason=StopReason.STOP,
        timestamp=timestamp,
    )


def test_ignores_stale_assistant_usage_after_a_newer_message_is_inserted_before_it() -> None:
    context = normalize_context(
        Context(
            system_prompt="system",
            messages=[
                UserMessage(content="summary", timestamp=200),
                create_assistant(100, 9_500),
                UserMessage(content="x" * 4_000, timestamp=300),
            ],
        )
    )

    estimate = estimate_context_tokens(context)
    assert estimate.tokens == 1_005
    assert estimate.usage_tokens == 0
    assert estimate.trailing_tokens == 1_005
    assert estimate.last_usage_index is None
    assert build_base_options(MODEL, context).max_tokens == 4_899


def test_uses_assistant_usage_again_after_a_response_to_the_inserted_context() -> None:
    context = normalize_context(
        Context(
            messages=[
                UserMessage(content="summary", timestamp=200),
                create_assistant(100, 9_500),
                UserMessage(content="new prompt", timestamp=300),
                create_assistant(400, 2_000),
                UserMessage(content="tail", timestamp=500),
            ]
        )
    )

    estimate = estimate_context_tokens(context)
    assert estimate.tokens == 2_001
    assert estimate.usage_tokens == 2_000
    assert estimate.trailing_tokens == 1
    assert estimate.last_usage_index == 3
