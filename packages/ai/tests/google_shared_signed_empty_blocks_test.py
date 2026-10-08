"""移植 ``packages/ai/test/google-shared-signed-empty-blocks.test.ts``。

Gemini 会给可见文本为空的响应 part（例如函数调用前的思考片段）附上
``thoughtSignature``，并要求下次请求原样回传该签名。空的 text/thinking 块
只有在*未签名*时才会被跳过。
"""

from __future__ import annotations

import json
from dataclasses import replace

from pi_ai.api.google_shared import convert_messages
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
    UserMessage,
)
from pi_ai.utils.transcript import normalize_context

VALID_SIG = "AAAAAAAAAAAAAAAAAAAAAA=="


def make_model(id: str = "gemini-3-pro-preview") -> Model:
    return Model(
        id=id,
        name=id,
        api="google-generative-ai",
        provider="google",
        base_url="https://example.com",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=8192,
    )


def make_context(model: Model, content: list) -> Context:
    return Context(
        messages=[
            UserMessage(content="Hi", timestamp=0),
            AssistantMessage(
                api=model.api,
                provider=model.provider,
                model=model.id,
                content=content,
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=0,
            ),
        ]
    )


def _model_turn(contents: list[dict]) -> dict:
    model_turn = next((content for content in contents if content.get("role") == "model"), None)
    assert model_turn is not None
    return model_turn


def test_keeps_a_signed_empty_thinking_block_so_its_signature_is_echoed_back() -> None:
    model = make_model()
    contents = convert_messages(
        model,
        normalize_context(
            make_context(
                model,
                [
                    ThinkingContent(thinking="", thinking_signature=VALID_SIG),
                    ToolCall(id="call_1", name="bash", arguments={"command": "ls"}),
                ],
            )
        ),
    )
    model_turn = _model_turn(contents)
    signed = [part for part in (model_turn.get("parts") or []) if part.get("thoughtSignature") == VALID_SIG]

    assert len(signed) == 1
    assert signed[0].get("thought") is True


def test_keeps_a_signed_empty_text_block_the_same_way() -> None:
    model = make_model()
    contents = convert_messages(
        model,
        normalize_context(
            make_context(
                model,
                [
                    TextContent(text="", text_signature=VALID_SIG),
                    ToolCall(id="call_1", name="bash", arguments={"command": "ls"}),
                ],
            )
        ),
    )
    model_turn = _model_turn(contents)
    signed = [part for part in (model_turn.get("parts") or []) if part.get("thoughtSignature") == VALID_SIG]

    assert len(signed) == 1


def test_still_drops_unsigned_empty_blocks() -> None:
    model = make_model()
    contents = convert_messages(
        model,
        normalize_context(
            make_context(
                model,
                [
                    ThinkingContent(thinking=""),
                    TextContent(text="   "),
                    ToolCall(id="call_1", name="bash", arguments={"command": "ls"}),
                ],
            )
        ),
    )
    model_turn = _model_turn(contents)

    assert len(model_turn.get("parts") or []) == 1
    assert model_turn["parts"][0].get("functionCall")


def test_still_drops_signed_empty_blocks_from_a_different_provider_or_model() -> None:
    model = make_model()
    contents = convert_messages(
        model,
        normalize_context(
            make_context(
                replace(model, id="other-model"),
                [
                    ThinkingContent(thinking="", thinking_signature=VALID_SIG),
                    TextContent(text="", text_signature=VALID_SIG),
                    ToolCall(id="call_1", name="bash", arguments={"command": "ls"}),
                ],
            )
        ),
    )
    model_turn = _model_turn(contents)

    assert len(model_turn.get("parts") or []) == 1
    assert model_turn["parts"][0].get("functionCall")
    assert VALID_SIG not in json.dumps(model_turn)
