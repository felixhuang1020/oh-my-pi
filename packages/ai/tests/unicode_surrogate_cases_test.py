"""移植自 ``packages/ai/test/unicode-surrogate.test.ts``。

provider 矩阵为真实 E2E，以带说明的跳过用例保留。文件靠后的本地子集将上游三个
tool-result 夹具（emoji、真实 LinkedIn 数据、未配对的高位代理项）对照本移植的
:func:`pi_ai.utils.sanitize_unicode.sanitize_surrogates` 测试——这正是真实用例在序列化
请求前所依赖的机制。
"""

from __future__ import annotations

import re

import pytest


def _slug(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "-", value).strip("-").lower()


def _case(group: str, test_name: str, gate: str) -> pytest.ParameterSet:
    return pytest.param(
        group,
        test_name,
        gate,
        id=f"{_slug(group)}--{_slug(test_name)}",
        marks=pytest.mark.skip(
            reason=(
                "live provider E2E: upstream gates on "
                f"{gate}; the Python suite is offline"
            )
        ),
    )


CASES = [
    _case('Google Provider Unicode Handling', 'should handle emoji in tool results', 'GEMINI_API_KEY'),
    _case('Google Provider Unicode Handling', 'should handle real-world LinkedIn comment data with emoji', 'GEMINI_API_KEY'),
    _case('Google Provider Unicode Handling', 'should handle unpaired high surrogate (0xD83D) in tool results', 'GEMINI_API_KEY'),
    _case('OpenAI Responses Provider Unicode Handling', 'should handle emoji in tool results', 'OPENAI_API_KEY'),
    _case('OpenAI Responses Provider Unicode Handling', 'should handle real-world LinkedIn comment data with emoji', 'OPENAI_API_KEY'),
    _case('OpenAI Responses Provider Unicode Handling', 'should handle unpaired high surrogate (0xD83D) in tool results', 'OPENAI_API_KEY'),
    _case('Anthropic Provider Unicode Handling', 'should handle emoji in tool results', 'ANTHROPIC_API_KEY'),
    _case('Anthropic Provider Unicode Handling', 'should handle real-world LinkedIn comment data with emoji', 'ANTHROPIC_API_KEY'),
    _case('Anthropic Provider Unicode Handling', 'should handle unpaired high surrogate (0xD83D) in tool results', 'ANTHROPIC_API_KEY'),
    _case('MiniMax Provider Unicode Handling', 'should handle emoji in tool results', 'MINIMAX_API_KEY'),
    _case('MiniMax Provider Unicode Handling', 'should handle real-world LinkedIn comment data with emoji', 'MINIMAX_API_KEY'),
    _case('MiniMax Provider Unicode Handling', 'should handle unpaired high surrogate (0xD83D) in tool results', 'MINIMAX_API_KEY'),
    _case('Xiaomi MiMo (API billing) Provider Unicode Handling', 'should handle emoji in tool results', 'XIAOMI_API_KEY'),
    _case('Xiaomi MiMo (API billing) Provider Unicode Handling', 'should handle real-world LinkedIn comment data with emoji', 'XIAOMI_API_KEY'),
    _case('Xiaomi MiMo (API billing) Provider Unicode Handling', 'should handle unpaired high surrogate (0xD83D) in tool results', 'XIAOMI_API_KEY'),
]


@pytest.mark.parametrize("group,test_name,gate", CASES)
def test_unicode_surrogate_live_case(group: str, test_name: str, gate: str) -> None:
    """上游每个用例都会发起一次真实 provider 请求。"""
    raise AssertionError("live provider E2E must not run")


# --------------------------------------------------------------------------------------
# 本地子集：真实用例在序列化请求前所依赖的净化器
# --------------------------------------------------------------------------------------

#: 与上游完全一致的夹具：来自 ``testEmojiInToolResults`` 的 emoji tool result。
EMOJI_TOOL_RESULT = """Test with emoji \U0001f648 and other characters:
- Monkey emoji: \U0001f648
- Thumbs up: \U0001f44d
- Heart: \u2764\ufe0f
- Thinking face: \U0001f914
- Rocket: \U0001f680
- Mixed text: Mario Zechner wann? Wo? Bin grad \u00e4u\u00dfersr eventuninformiert \U0001f648
- Japanese: \u3053\u3093\u306b\u3061\u306f
- Chinese: \u4f60\u597d
- Mathematical symbols: \u2211\u222b\u2202\u221a
- Special quotes: \u201ccurly\u201d \u2018quotes\u2019"""

#: 与上游完全一致的夹具：来自 ``testRealWorldLinkedInData`` 的真实 LinkedIn tool result。
LINKEDIN_TOOL_RESULT = """Post: Hab einen "Generative KI f\u00fcr Nicht-Techniker" Workshop gebaut.
Unanswered Comments: 2

=> {
  "comments": [
    {
      "author": "Matthias Neumayer's  graphic link",
      "text": "Leider nehmen das viel zu wenige Leute ernst"
    },
    {
      "author": "Matthias Neumayer's  graphic link",
      "text": "Mario Zechner wann? Wo? Bin grad \u00e4u\u00dfersr eventuninformiert \U0001f648"
    }
  ]
}"""


def test_preserves_valid_emoji_in_tool_results() -> None:
    from pi_ai.utils.sanitize_unicode import sanitize_surrogates

    assert sanitize_surrogates(EMOJI_TOOL_RESULT) == EMOJI_TOOL_RESULT
    assert "\U0001f648" in sanitize_surrogates(EMOJI_TOOL_RESULT)


def test_preserves_real_world_linkedin_comment_data_with_emoji() -> None:
    from pi_ai.utils.sanitize_unicode import sanitize_surrogates

    assert sanitize_surrogates(LINKEDIN_TOOL_RESULT) == LINKEDIN_TOOL_RESULT
    assert "\U0001f648" in sanitize_surrogates(LINKEDIN_TOOL_RESULT)


def test_sanitizes_an_unpaired_high_surrogate_in_tool_results() -> None:
    from pi_ai.utils.sanitize_unicode import sanitize_surrogates

    unpaired = chr(0xD83D)  # 缺少配对低位代理项的高位代理项
    text = f"Text with unpaired surrogate: {unpaired} <- should be sanitized"

    sanitized = sanitize_surrogates(text)

    assert unpaired not in sanitized
    assert sanitized == "Text with unpaired surrogate:  <- should be sanitized"
    assert not any("\ud800" <= char <= "\udfff" for char in sanitized)


def test_sanitizes_a_lone_low_surrogate_too() -> None:
    from pi_ai.utils.sanitize_unicode import sanitize_surrogates

    unpaired = chr(0xDC00)  # 缺少配对高位代理项的低位代理项
    assert sanitize_surrogates(f"before {unpaired} after") == "before  after"
