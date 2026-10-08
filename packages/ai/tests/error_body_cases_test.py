"""移植 ``packages/ai/test/error-body.test.ts``。

共享的 provider 错误体归一化器（:mod:`pi_ai.utils.error_body`）单元测试。
上游手工构造 SDK 形态的错误对象；这里用普通 ``Exception`` 加 ``setattr``
复现同样的属性结构（``status``、``error``、``$metadata``/``$response``、
类 readable-stream 对象，以及不应被序列化的类实例）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from pi_ai.utils.error_body import (
    MAX_PROVIDER_ERROR_BODY_CHARS,
    format_provider_error,
    normalize_provider_error,
)

#: ``JSON.stringify`` 输出紧凑 JSON；上游精确字符串 fixture 依赖该格式。
_JSON_SEPARATORS = (",", ":")


def _json(value: Any) -> str:
    """按 ``JSON.stringify`` 的方式序列化，以匹配上游精确字符串 fixture。"""
    return json.dumps(value, ensure_ascii=False, separators=_JSON_SEPARATORS)


def _make_error(message: str, **attributes: Any) -> Exception:
    """等价于 ``Object.assign(new Error(message), attributes)``。"""
    error = Exception(message)
    for name, value in attributes.items():
        setattr(error, name, value)
    return error


class _SdkHttpResponseBody:
    """类实例形式的响应体：不是普通对象，因此必须被忽略。"""

    def __init__(self) -> None:
        self.locked = False
        self.state = {"storedError": None}


class _SdkInnerError:
    """类实例形式的 ``error`` 字段：不是普通对象，因此必须被忽略。"""

    def __init__(self) -> None:
        self.code = "EPROTO"
        self.internalState: dict[str, Any] = {}


# ---------------------------------------------------------------------------------
# 测试 normalizeProviderError
# ---------------------------------------------------------------------------------


def test_extracts_status_and_body_from_a_mistral_shaped_error():
    error = _make_error(
        "Mistral request failed",
        status_code=403,
        body='{"error":"blocked by gateway WAF"}',
    )

    norm = normalize_provider_error(error)

    assert norm.status == 403
    assert norm.body == '{"error":"blocked by gateway WAF"}'
    assert norm.message_carries_body is False


def test_reads_the_parsed_body_off_an_openai_api_error_when_the_message_is_opaque():
    # makeMessage(status, error, message) 在解析体未解析时会生成
    # "<status> status code (no body)"，而 body 仍保留在 error.error 上。
    error = _make_error(
        "403 status code (no body)",
        status=403,
        error={"error": "blocked by gateway WAF"},
    )

    norm = normalize_provider_error(error)

    assert norm.status == 403
    assert norm.body == '{"error":"blocked by gateway WAF"}'
    assert norm.message_carries_body is False


def test_preserves_the_message_when_google_genai_already_folds_the_body_into_it():
    body = {"error": {"code": 403, "message": "Permission denied"}}
    error = _make_error(_json(body), status=403)

    norm = normalize_provider_error(error)

    assert norm.status == 403
    assert norm.message_carries_body is True
    assert norm.message == _json(body)


def test_extracts_status_and_body_from_a_bedrock_shaped_service_exception():
    error = _make_error(
        "UnknownError",
        name="UnknownError",
        **{
            "$metadata": {"httpStatusCode": 403},
            "$response": SimpleNamespace(statusCode=403, body='{"message":"blocked by gateway WAF"}'),
        },
    )

    norm = normalize_provider_error(error)

    assert norm.status == 403
    assert norm.body == '{"message":"blocked by gateway WAF"}'
    assert norm.message_carries_body is False


def test_ignores_a_bedrock_response_stream_instead_of_serializing_its_internals():
    error = _make_error(
        "Invocation of model ID anthropic.claude-opus-5 with on-demand throughput isn't supported.",
        name="ValidationException",
        **{
            "$metadata": {"httpStatusCode": 400},
            "$response": SimpleNamespace(
                statusCode=400,
                body=SimpleNamespace(pipe=lambda *args: None, _events={"close": [None, None]}),
            ),
        },
    )

    norm = normalize_provider_error(error)

    assert norm.status == 400
    assert norm.body is None
    assert "on-demand throughput isn't supported" in norm.message
    assert norm.message_carries_body is True


def test_ignores_a_class_instance_response_body_without_a_pipe_method_instead_of_serializing_it():
    # 并非所有 SDK 响应包装都是 node 流：web ReadableStream 和 SDK 专有包装类
    # 没有 `pipe`，但序列化它们仍会产出覆盖真实消息的内部结构噪音。
    error = _make_error(
        "Input is too long for requested model.",
        name="ValidationException",
        **{
            "$metadata": {"httpStatusCode": 400},
            "$response": SimpleNamespace(statusCode=400, body=_SdkHttpResponseBody()),
        },
    )

    norm = normalize_provider_error(error)

    assert norm.status == 400
    assert norm.body is None
    assert "Input is too long" in norm.message
    assert norm.message_carries_body is True


def test_ignores_a_class_instance_error_field_instead_of_serializing_it():
    error = _make_error("TLS handshake failed", status=502, error=_SdkInnerError())

    norm = normalize_provider_error(error)

    assert norm.body is None
    assert norm.message == "TLS handshake failed"
    assert norm.message_carries_body is True


def test_still_surfaces_a_plain_parsed_json_body_object():
    error = _make_error(
        "400 status code (no body)",
        status=400,
        error={"message": "schema validation failed", "field": "tools[0]"},
    )

    norm = normalize_provider_error(error)

    assert norm.body == '{"message":"schema validation failed","field":"tools[0]"}'
    assert norm.message_carries_body is False


def test_json_stringifies_a_non_error_thrown_value():
    norm = normalize_provider_error({"reason": "boom"})

    assert norm.status is None
    assert norm.body is None
    assert norm.message == '{"reason":"boom"}'
    assert norm.message_carries_body is False


def test_treats_an_empty_parsed_body_object_as_no_body():
    error = _make_error("403 status code (no body)", status=403, error={})

    norm = normalize_provider_error(error)

    assert norm.body is None
    assert norm.message_carries_body is True


def test_truncates_the_body_at_the_cap():
    long_body = "x" * (MAX_PROVIDER_ERROR_BODY_CHARS + 50)
    error = _make_error("failed", status_code=500, body=long_body)

    norm = normalize_provider_error(error)

    assert "... [truncated 50 chars]" in norm.body
    assert len(norm.body) < len(long_body)


def test_sets_message_carries_body_when_the_message_already_contains_the_extracted_body():
    error = _make_error("500: upstream exploded", status_code=500, body="upstream exploded")

    norm = normalize_provider_error(error)

    assert norm.message_carries_body is True


# ---------------------------------------------------------------------------------
# 测试 formatProviderError
# ---------------------------------------------------------------------------------


def test_format_surfaces_status_and_body_without_a_prefix():
    norm = normalize_provider_error(
        _make_error(
            "403 status code (no body)",
            status=403,
            error={"error": "blocked by gateway WAF"},
        )
    )

    formatted = format_provider_error(norm)

    assert "403" in formatted
    assert "blocked by gateway WAF" in formatted
    assert formatted != "403 status code (no body)"


def test_format_applies_a_provider_prefix_with_status_and_body():
    norm = normalize_provider_error(
        _make_error(
            "403 status code (no body)",
            status=403,
            error={"error": "blocked by gateway WAF"},
        )
    )

    assert (
        format_provider_error(norm, "OpenAI API error")
        == 'OpenAI API error (403): {"error":"blocked by gateway WAF"}'
    )


def test_format_preserves_the_message_with_prefix_and_status_when_it_already_carries_the_body():
    body = _json({"error": {"message": "Permission denied"}})
    norm = normalize_provider_error(_make_error(body, status=403))

    assert format_provider_error(norm, "OpenAI API error") == f"OpenAI API error (403): {body}"


def test_format_returns_the_bare_message_for_a_non_error_value():
    norm = normalize_provider_error({"reason": "boom"})

    assert format_provider_error(norm) == '{"reason":"boom"}'
