"""针对 provider payload 的 Unicode 清洗."""

from __future__ import annotations

import re

__all__ = ["sanitize_surrogates"]

_UNPAIRED_SURROGATE = re.compile(
    "[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]"
)


def sanitize_surrogates(text: str) -> str:
    r"""移除 ``text`` 中孤立的 UTF-16 代理项（surrogate）码元.

    孤立代理项会导致许多 API provider 的 JSON 序列化失败；正确成对的代理项
    （合法 emoji 等增补平面字符）会被原样保留。

    Args:
        text: 待清洗的字符串。

    Returns:
        移除孤立代理项后的字符串。

    Examples:
        >>> paired = chr(0x1F648)                     # 一个合法的增补平面 emoji
        >>> sanitize_surrogates(f"Hello {paired}") == f"Hello {paired}"
        True
        >>> sanitize_surrogates(f"Text {chr(0xD83D)} here")
        'Text  here'
    """
    return _UNPAIRED_SURROGATE.sub("", text)
