"""移植自 ``openai-responses-empty-tool-result.test.ts``。"""

from __future__ import annotations

from pi_ai.api.openai_responses_shared import convert_responses_messages
from pi_ai.compat import get_model
from pi_ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage

from openai_port_harness import responses_context

_ALLOWED_PROVIDERS = {"openai", "openai-codex", "opencode"}


async def test_uses_no_tool_output_placeholder_for_empty_tool_results_without_images():
    model = get_model("openai", "gpt-4o-mini")
    now = 1000
    assistant = AssistantMessage(
        role="assistant",
        content=[ToolCall(type="toolCall", id="tool-1", name="bash", arguments={"command": "true"})],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=now,
    )
    empty_tool_result = ToolResultMessage(
        role="toolResult",
        tool_call_id="tool-1",
        tool_name="bash",
        content=[TextContent(type="text", text="")],
        is_error=False,
        timestamp=now + 1,
    )

    context = responses_context(
        [
            UserMessage(content="Run the command", timestamp=now - 1),
            assistant,
            empty_tool_result,
        ]
    )

    items = convert_responses_messages(model, context, _ALLOWED_PROVIDERS)
    function_call_output = next(
        (item for item in items if item.get("type") == "function_call_output"), None
    )

    assert function_call_output is not None
    assert function_call_output["output"] == "(no tool output)"
    assert "see attached image" not in function_call_output["output"]
