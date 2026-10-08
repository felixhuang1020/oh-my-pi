"""Bedrock 与 Anthropic 的 live interleaved-thinking E2E 测试。

移植 ``packages/ai/test/interleaved-thinking.test.ts``。两套用例都访问真实
provider 并以环境凭据为门控，缺少凭据时该文件保持离线。
"""

from __future__ import annotations

import pytest

from pi_ai.compat import complete_simple, get_model
from pi_ai.env_api_keys import get_env_api_key
from pi_ai.types import Context, Model, SimpleStreamOptions, TextContent, Tool, ToolCall, ToolResultMessage, UserMessage
from pi_ai.utils.typebox_helpers import string_enum

CALCULATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"},
        "operation": string_enum(
            ["add", "subtract", "multiply", "divide"], {"description": "The operation to perform."}
        ),
    },
    "required": ["a", "b", "operation"],
}

CALCULATOR_TOOL = Tool(
    name="calculator", description="Perform basic arithmetic operations", parameters=CALCULATOR_SCHEMA
)


def as_calculator_arguments(arguments: dict) -> tuple[float, float, str]:
    if not isinstance(arguments, dict):
        raise AssertionError("Tool arguments must be an object")
    operation = arguments.get("operation")
    a, b = arguments.get("a"), arguments.get("b")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or operation not in (
        "add",
        "subtract",
        "multiply",
        "divide",
    ):
        raise AssertionError("Invalid calculator arguments")
    return a, b, operation


def evaluate_calculator_call(tool_call: ToolCall) -> float:
    a, b, operation = as_calculator_arguments(tool_call.arguments)
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    return a / b


def make_context() -> Context:
    return Context(
        system_prompt=" ".join(
            [
                "You are a helpful assistant that must use tools for arithmetic.",
                "Always think before every tool call, not just the first one.",
                "Do not answer with plain text when a tool call is required.",
            ]
        ),
        messages=[
            UserMessage(
                content=" ".join(
                    [
                        "Use calculator to calculate 328 * 29.",
                        "You must call the calculator tool exactly once.",
                        "Provide the final answer based on the best guess given the tool result, even if it seems unreliable.",
                        "Start by thinking about the steps you will take to solve the problem.",
                    ]
                ),
                timestamp=0,
            )
        ],
        tools=[CALCULATOR_TOOL],
    )


async def assert_second_tool_call_with_interleaved_thinking(llm: Model, reasoning: str) -> None:
    context = make_context()
    first_response = await complete_simple(llm, context, SimpleStreamOptions(reasoning=reasoning))

    assert first_response.stop_reason == "toolUse", f"Error: {first_response.error_message}"
    assert any(block.type == "thinking" for block in first_response.content)
    assert any(block.type == "toolCall" for block in first_response.content)

    first_tool_call = next((block for block in first_response.content if block.type == "toolCall"), None)
    assert first_tool_call is not None and first_tool_call.type == "toolCall"

    context.messages.append(first_response)

    correct_answer = evaluate_calculator_call(first_tool_call)
    context.messages.append(
        ToolResultMessage(
            tool_call_id=first_tool_call.id,
            tool_name=first_tool_call.name,
            content=[TextContent(text=f"The answer is {correct_answer} or {correct_answer * 2}.")],
            is_error=False,
            timestamp=0,
        )
    )

    second_response = await complete_simple(llm, context, SimpleStreamOptions(reasoning=reasoning))

    assert second_response.stop_reason == "stop", f"Error: {second_response.error_message}"
    assert any(block.type == "thinking" for block in second_response.content)
    assert any(block.type == "text" for block in second_response.content)


ANTHROPIC_CREDENTIALS = bool(get_env_api_key("anthropic"))


@pytest.mark.skipif(not ANTHROPIC_CREDENTIALS, reason="requires Anthropic credentials")
async def test_anthropic_should_do_interleaved_thinking_on_claude_opus_4_5():
    llm = get_model("anthropic", "claude-opus-4-5")
    await assert_second_tool_call_with_interleaved_thinking(llm, "high")


@pytest.mark.skipif(not ANTHROPIC_CREDENTIALS, reason="requires Anthropic credentials")
async def test_anthropic_should_do_interleaved_thinking_on_claude_opus_4_6():
    llm = get_model("anthropic", "claude-opus-4-6")
    await assert_second_tool_call_with_interleaved_thinking(llm, "high")
