"""移植 ``packages/ai/test/google-thinking-signature.test.ts``。"""

from __future__ import annotations

from pi_ai.api.google_shared import is_thinking_part, retain_thought_signature


def test_treats_part_thought_true_as_thinking() -> None:
    assert is_thinking_part({"thought": True, "thoughtSignature": None}) is True
    assert is_thinking_part({"thought": True, "thoughtSignature": "opaque-signature"}) is True


def test_does_not_treat_thought_signature_alone_as_thinking() -> None:
    # 按 Google 文档，thoughtSignature 用于上下文回放，可出现在任意 part 类型上；
    # 只有 thought === true 才表示 thinking 内容。
    # 参见: https://ai.google.dev/gemini-api/docs/thought-signatures
    assert is_thinking_part({"thought": None, "thoughtSignature": "opaque-signature"}) is False
    assert is_thinking_part({"thought": False, "thoughtSignature": "opaque-signature"}) is False


def test_does_not_treat_empty_or_missing_signatures_as_thinking_if_thought_is_not_set() -> None:
    assert is_thinking_part({"thought": None, "thoughtSignature": None}) is False
    assert is_thinking_part({"thought": False, "thoughtSignature": ""}) is False


def test_preserves_the_existing_signature_when_subsequent_deltas_omit_thought_signature() -> None:
    first = retain_thought_signature(None, "sig-1")
    assert first == "sig-1"

    second = retain_thought_signature(first, None)
    assert second == "sig-1"

    third = retain_thought_signature(second, "")
    assert third == "sig-1"


def test_updates_the_signature_when_a_new_non_empty_signature_arrives() -> None:
    updated = retain_thought_signature("sig-1", "sig-2")
    assert updated == "sig-2"
