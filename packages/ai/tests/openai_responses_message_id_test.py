"""移植自 ``openai-responses-message-id.test.ts``。"""

from __future__ import annotations

from pi_ai.api.openai_responses_shared import convert_responses_messages
from pi_ai.compat import get_model
from pi_ai.types import AssistantMessage, TextContent, ThinkingContent, Usage, UserMessage

from openai_port_harness import responses_context

_ALLOWED_PROVIDERS = {"openai"}


def test_generates_unique_fallback_message_ids_for_multiple_text_blocks_in_one_assistant_turn():
    model = get_model("openai", "gpt-5.5")
    assistant = AssistantMessage(
        role="assistant",
        content=[
            ThinkingContent(type="thinking", thinking="private reasoning"),
            TextContent(type="text", text="visible answer"),
        ],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-opus-4-8",
        usage=Usage(),
        stop_reason="stop",
        timestamp=1000,
    )
    context = responses_context(
        [UserMessage(content="hello", timestamp=0), assistant],
        system_prompt="You are concise.",
    )

    items = convert_responses_messages(model, context, _ALLOWED_PROVIDERS)
    message_ids = [
        item["id"]
        for item in items
        if item.get("type") == "message" and isinstance(item.get("id"), str)
    ]

    assert message_ids == ["msg_pi_1", "msg_pi_1_1"]
    assert len(set(message_ids)) == len(message_ids)
