"""移植 ``packages/ai/test/google-shared-gemini3-unsigned-tool-call.test.ts``。"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from pi_ai.api.google_shared import convert_messages, requires_tool_call_id
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_ai.utils.transcript import normalize_context


def make_gemini3_model(
    api: str,
    provider: str,
    id: str = "gemini-3-pro-preview",
) -> Model:
    return Model(
        id=id,
        name="Gemini 3 Pro Preview",
        api=api,
        provider=provider,
        base_url="https://example.com",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=8192,
    )


def make_context(model: Model, thought_signature: str | None = None) -> Context:
    return Context(
        messages=[
            UserMessage(content="Hi", timestamp=0),
            AssistantMessage(
                api=model.api,
                provider=model.provider,
                model=model.id,
                content=[
                    ToolCall(
                        id="call_1",
                        name="bash",
                        arguments={"command": "echo hi"},
                        thought_signature=thought_signature,
                    ),
                    ToolCall(id="call_2", name="bash", arguments={"command": "ls -la"}),
                ],
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=0,
            ),
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="bash",
                content=[TextContent(text="hi")],
                is_error=False,
                timestamp=0,
            ),
            ToolResultMessage(
                tool_call_id="call_2",
                tool_name="bash",
                content=[TextContent(text="files")],
                is_error=False,
                timestamp=0,
            ),
        ]
    )


def _parts(contents: list[dict]) -> list[dict]:
    return [part for content in contents for part in (content.get("parts") or [])]


@pytest.mark.parametrize(
    ("api", "provider", "model_id"),
    [
        ("google-generative-ai", "google", "gemini-3-pro-preview"),
        ("google-generative-ai", "google", "gemini-3.6-flash"),
        ("google-vertex", "google-vertex", "gemini-3-pro-preview"),
    ],
)
def test_preserves_tool_call_ids_for_gemini3_history(api: str, provider: str, model_id: str) -> None:
    model = make_gemini3_model(api, provider, model_id)
    contents = convert_messages(model, normalize_context(make_context(model)))

    function_call_ids = [
        part["functionCall"]["id"]
        for part in _parts(contents)
        if part.get("functionCall") and part["functionCall"].get("id")
    ]
    function_response_ids = [
        part["functionResponse"]["id"]
        for part in _parts(contents)
        if part.get("functionResponse") and part["functionResponse"].get("id")
    ]

    assert function_call_ids == ["call_1", "call_2"]
    assert function_response_ids == ["call_1", "call_2"]


def test_does_not_add_skip_thought_signature_validator_for_unsigned_google_gen_ai_tool_calls() -> None:
    model = make_gemini3_model("google-generative-ai", "google")
    contents = convert_messages(model, normalize_context(make_context(replace(model, id="other-model"))))

    model_turn = next((content for content in contents if content.get("role") == "model"), None)
    assert model_turn is not None

    function_call_parts = [part for part in (model_turn.get("parts") or []) if part.get("functionCall") is not None]
    assert len(function_call_parts) == 2
    assert function_call_parts[0].get("thoughtSignature") is None
    assert function_call_parts[1].get("thoughtSignature") is None
    assert "skip_thought_signature_validator" not in json.dumps(model_turn)

    text_parts = [part for part in (model_turn.get("parts") or []) if part.get("text") is not None]
    historical_text = [part for part in text_parts if "Historical context" in (part.get("text") or "")]
    assert historical_text == []


def test_does_not_add_skip_thought_signature_validator_for_unsigned_vertex_tool_calls() -> None:
    model = make_gemini3_model("google-vertex", "google-vertex")
    contents = convert_messages(model, normalize_context(make_context(model)))

    model_turn = next((content for content in contents if content.get("role") == "model"), None)
    assert model_turn is not None
    function_call_parts = [part for part in (model_turn.get("parts") or []) if part.get("functionCall") is not None]

    assert len(function_call_parts) == 2
    assert function_call_parts[0].get("thoughtSignature") is None
    assert function_call_parts[1].get("thoughtSignature") is None
    assert "skip_thought_signature_validator" not in json.dumps(model_turn)


def test_preserves_valid_thought_signature_when_present_for_the_same_provider_and_model() -> None:
    model = make_gemini3_model("google-generative-ai", "google")
    valid_sig = "AAAAAAAAAAAAAAAAAAAAAA=="
    contents = convert_messages(model, normalize_context(make_context(model, valid_sig)))

    model_turn = next((content for content in contents if content.get("role") == "model"), None)
    assert model_turn is not None
    function_call_parts = [part for part in (model_turn.get("parts") or []) if part.get("functionCall") is not None]

    assert len(function_call_parts) == 2
    assert function_call_parts[0].get("thoughtSignature") == valid_sig
    assert function_call_parts[1].get("thoughtSignature") is None


def test_does_not_add_a_thought_signature_for_non_gemini_3_models() -> None:
    model = make_gemini3_model("google-generative-ai", "google", "gemini-2.5-flash")
    contents = convert_messages(model, normalize_context(make_context(replace(model, id="other-model"))))

    model_turn = next((content for content in contents if content.get("role") == "model"), None)
    assert model_turn is not None
    function_call_parts = [part for part in (model_turn.get("parts") or []) if part.get("functionCall") is not None]
    function_response_parts = [part for part in _parts(contents) if part.get("functionResponse") is not None]

    assert len(function_call_parts) == 2
    assert all(part["functionCall"].get("id") is None for part in function_call_parts)
    assert all(part.get("thoughtSignature") is None for part in function_call_parts)
    assert len(function_response_parts) == 2
    assert all(part["functionResponse"].get("id") is None for part in function_response_parts)


@pytest.mark.parametrize(
    ("expected", "model_id"),
    [
        (False, "gemini-2.5-flash"),
        (True, "gemini-3.6-flash"),
        (True, "claude-sonnet-4-5"),
        (True, "gpt-oss-120b"),
    ],
)
def test_requires_tool_call_id(expected: bool, model_id: str) -> None:
    assert requires_tool_call_id(model_id) is expected
