"""移植 ``packages/ai/test/lax-message-content.test.ts``。

Message 类型要求 ``content`` 始终存在，但无类型的调用方（自定义工具、手工拼的
历史、旧会话文件）可能违反该约定。``transform_messages`` 是每次 provider 请求前
的必经点，且有意保持宽松：把 null/缺失的 content 归一化为空数组
（issues #6259、#6276）。
"""

from __future__ import annotations

from pi_ai.api.transform_messages import transform_messages
from pi_ai.types import (
    AssistantMessage,
    Model,
    ModelCost,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def make_text_only_model() -> Model:
    # 纯文本模型，使图片降级路径（replaceImagesWithPlaceholder）被触发——
    # 那是 null tool result content 的主要崩溃点。
    return Model(
        id="test-model",
        name="Test Model",
        api="openai-completions",
        provider="openai",
        base_url="https://example.invalid/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=16000,
    )


def test_normalizes_null_missing_content_to_an_empty_array_instead_of_crashing() -> None:
    messages = [
        UserMessage(content=None, timestamp=0),
        AssistantMessage(
            content=None,
            api="openai-completions",
            provider="openai",
            model="test-model",
            usage=Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0),
            stop_reason="stop",
            timestamp=0,
        ),
        ToolResultMessage(
            tool_call_id="call_1",
            tool_name="web_search",
            is_error=False,
            timestamp=0,
        ),
    ]

    result = transform_messages(messages, make_text_only_model())

    assert len(result) == 3
    for message in result:
        assert message.content == []
