"""移植自 ``openai-responses-foreign-toolcall-id.test.ts``。"""

from __future__ import annotations

import re

from pi_ai.api.openai_responses_shared import convert_responses_messages
from pi_ai.compat import get_model
from pi_ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage
from pi_ai.utils.hash import short_hash

from openai_port_harness import responses_context

FOREIGN_RAW_TOOL_CALL_ID = (
    "call_4VnzVawQXPB9MgYib7CiQFEY|I9b95oN1wD/cHXKTw3PpRkL6KkCtzTJhUxMouMWYwHeTo2j3htzfSk7YPx2vifiIM4g3A8XXyOj8q4Bt6SLUG7gqY1E3ELkrkVQNHglRfUmWj84lqxJY+Puieb3VKyX0FB+83TUzn91cDMF/4gzt990IzqVrc+nIb9RRscRD070Du16q1glydVjWR0SBJsE6TbY/esOjFpqplogQqrajm1eI++f3eLi73R6q7hVusY0QbeFySVxABCjhN0lXB04caBe1rzHjYzul6MAXj7uq+0r17VLq+yrtyYhN12wkmFqHeqTyEei6EFPbMy24Nc+IbJlkP0OCg02W+gOnyBFcbi2ctvJFSOhSjt1CqBdqCnnhwUqXjbWiT0wh3DmLScRgTHmGkaI+oAcQQjfic65nxj+TnEkReA=="
)

_ALLOWED_PROVIDERS = {"openai"}


def test_hashes_foreign_tool_item_ids_into_a_bounded_responses_safe_shape():
    model = get_model("openai", "gpt-5.5")
    assistant = AssistantMessage(
        role="assistant",
        content=[
            ToolCall(
                type="toolCall",
                id=FOREIGN_RAW_TOOL_CALL_ID,
                name="edit",
                arguments={"path": "src/styles/app.css"},
            )
        ],
        api="openai-responses",
        provider="anthropic",
        model="gpt-5.5",
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=1000,
    )
    tool_result = ToolResultMessage(
        role="toolResult",
        tool_call_id=FOREIGN_RAW_TOOL_CALL_ID,
        tool_name="edit",
        content=[TextContent(type="text", text="ok")],
        is_error=False,
        timestamp=2000,
    )
    context = responses_context(
        [
            UserMessage(content="Use the tool.", timestamp=0),
            assistant,
            tool_result,
        ],
        system_prompt="You are concise.",
    )

    items = convert_responses_messages(model, context, _ALLOWED_PROVIDERS)
    function_call = next((item for item in items if item.get("type") == "function_call"), None)

    assert function_call is not None
    expected_item_id = f"fc_{short_hash(FOREIGN_RAW_TOOL_CALL_ID.split('|')[1])}"
    assert function_call["id"] == expected_item_id
    assert len(function_call["id"]) <= 64
    assert re.match(r"^fc_[A-Za-z0-9]+$", function_call["id"])
