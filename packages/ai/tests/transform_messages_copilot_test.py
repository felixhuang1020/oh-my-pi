"""移植自 ``packages/ai/test/transform-messages-copilot-openai-to-anthropic.test.ts``。

GitHub Copilot Claude 的跨 provider 会话迁移：thinking 块转为纯文本、丢弃
``thoughtSignature``，并用合成 tool result 收尾孤立的末尾 tool call。
"""

from __future__ import annotations

import json
import re

from pi_ai.api.transform_messages import transform_messages
from pi_ai.types import (
    AssistantMessage,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)

# --------------------------------------------------------------------------------------
# 测试夹具（上游将这些辅助函数定义在测试文件中）
# --------------------------------------------------------------------------------------


def anthropic_normalize_tool_call_id(tool_call_id: str, _model: Model, _source: AssistantMessage) -> str:
    """``anthropic.ts`` 使用的 id 归一化：仅保留 ``[a-zA-Z0-9_-]`` 且最长 64 字符。"""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_call_id)[:64]


def make_copilot_claude_model() -> Model:
    return Model(
        id="claude-sonnet-4.6",
        name="Claude Sonnet 4.6",
        api="anthropic-messages",
        provider="github-copilot",
        base_url="https://api.individual.githubcopilot.com",
        reasoning=True,
        input=["text", "image"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=16000,
    )


def make_assistant_message(content: list) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=content,
        api="openai-responses",
        provider="github-copilot",
        model="gpt-5",
        usage=Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0),
        stop_reason="toolUse",
        timestamp=0,
    )


def _empty_usage() -> Usage:
    return Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0)


# --------------------------------------------------------------------------------------
# describe("OpenAI to Anthropic session migration for Copilot Claude")：OpenAI → Anthropic 会话迁移
# --------------------------------------------------------------------------------------


def test_converts_thinking_blocks_to_plain_text_when_source_model_differs() -> None:
    model = make_copilot_claude_model()
    messages = [
        UserMessage(content="hello", timestamp=0),
        AssistantMessage(
            role="assistant",
            content=[
                ThinkingContent(thinking="Let me think about this...", thinking_signature="reasoning_content"),
                TextContent(text="Hi there!"),
            ],
            api="openai-completions",
            provider="github-copilot",
            model="gpt-4o",
            usage=_empty_usage(),
            stop_reason="stop",
            timestamp=0,
        ),
    ]

    result = transform_messages(messages, model, anthropic_normalize_tool_call_id)
    assistant_msg = next(message for message in result if message.role == "assistant")

    # 模型不同，thinking 块应被转换为文本
    text_blocks = [block for block in assistant_msg.content if block.type == "text"]
    thinking_blocks = [block for block in assistant_msg.content if block.type == "thinking"]
    assert len(thinking_blocks) == 0
    assert len(text_blocks) >= 2


def test_removes_thought_signature_from_tool_calls_when_migrating_between_models() -> None:
    model = make_copilot_claude_model()
    messages = [
        UserMessage(content="run a command", timestamp=0),
        AssistantMessage(
            role="assistant",
            content=[
                ToolCall(
                    id="call_123",
                    name="bash",
                    arguments={"command": "ls"},
                    thought_signature=json.dumps(
                        {"type": "reasoning.encrypted", "id": "call_123", "data": "encrypted"}
                    ),
                ),
            ],
            api="openai-responses",
            provider="github-copilot",
            model="gpt-5",
            usage=_empty_usage(),
            stop_reason="toolUse",
            timestamp=0,
        ),
        ToolResultMessage(
            tool_call_id="call_123",
            tool_name="bash",
            content=[TextContent(text="output")],
            is_error=False,
            timestamp=0,
        ),
    ]

    result = transform_messages(messages, model, anthropic_normalize_tool_call_id)
    assistant_msg = next(message for message in result if message.role == "assistant")
    tool_call = next(block for block in assistant_msg.content if block.type == "toolCall")

    assert tool_call.thought_signature is None


def test_adds_synthetic_tool_results_for_trailing_orphaned_tool_calls() -> None:
    model = make_copilot_claude_model()
    messages = [
        UserMessage(content="read the file", timestamp=0),
        make_assistant_message(
            [
                ToolCall(id="call_123|fc_123", name="read", arguments={"path": "README.md"}),
            ]
        ),
    ]

    result = transform_messages(messages, model, anthropic_normalize_tool_call_id)
    last_message = result[-1]

    assert last_message.role == "toolResult"
    assert last_message.tool_call_id == "call_123_fc_123"
    assert last_message.tool_name == "read"
    assert last_message.is_error is True
    assert last_message.content == [TextContent(type="text", text="No result provided")]


def test_adds_synthetic_results_only_for_trailing_tool_calls_that_are_still_missing_results() -> None:
    model = make_copilot_claude_model()
    messages = [
        UserMessage(content="run commands", timestamp=0),
        make_assistant_message(
            [
                ToolCall(id="call_1|fc_1", name="read", arguments={"path": "README.md"}),
                ToolCall(id="call_2|fc_2", name="bash", arguments={"command": "pwd"}),
            ]
        ),
        ToolResultMessage(
            tool_call_id="call_1|fc_1",
            tool_name="read",
            content=[TextContent(text="done")],
            is_error=False,
            timestamp=0,
        ),
    ]

    result = transform_messages(messages, model, anthropic_normalize_tool_call_id)
    synthetic_results = [
        message for message in result if message.role == "toolResult" and message.is_error
    ]

    assert len(synthetic_results) == 1
    assert synthetic_results[0].role == "toolResult"
    assert synthetic_results[0].tool_call_id == "call_2_fc_2"
    assert synthetic_results[0].tool_name == "bash"
    assert synthetic_results[0].content == [TextContent(type="text", text="No result provided")]
