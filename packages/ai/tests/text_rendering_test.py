"""移植自 ``packages/ai/test/text.test.ts``。

覆盖 :func:`pi_ai.utils.text.content_text`：助手的 thinking/text/toolCall 块分别使用
默认与自定义分隔符的拼接、纯字符串直传，以及从 tool-result 内容块中提取文本。
"""

from __future__ import annotations

from pi_ai.types import TextContent, ThinkingContent, ToolCall
from pi_ai.utils.text import content_text

CONTENT = [
    ThinkingContent(thinking="reasoning"),
    TextContent(text="first"),
    ToolCall(id="1", name="read", arguments={}),
    TextContent(text="second"),
]


def test_extracts_assistant_text_blocks():
    assert content_text(CONTENT) == "first\nsecond"


def test_supports_custom_separators():
    assert content_text(CONTENT, "") == "firstsecond"


def test_passes_string_content_through():
    assert content_text("hello") == "hello"


def test_extracts_text_from_tool_result_content():
    tool_result_content = [
        TextContent(text="first"),
        TextContent(text="second"),
    ]

    assert content_text(tool_result_content, "") == "firstsecond"
