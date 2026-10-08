"""OpenAI prompt-cache key 截断。

移植自 ``packages/ai/src/api/openai-prompt-cache.ts``。OpenAI 拒绝超过 64 个字符的
``prompt_cache_key``，因此调用方需要在发送前截断 session id。
"""

from __future__ import annotations

__all__ = ["OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH", "clamp_openai_prompt_cache_key"]

#: ``prompt_cache_key`` 允许的最大长度。
OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH = 64


def clamp_openai_prompt_cache_key(key: str | None) -> str | None:
    """将 ``key`` 截断到 :data:`OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH` 个字符。

    计数方式与 TypeScript 的 ``Array.from(key)`` 一致：Python 同样按 Unicode 码点迭代，
    因此增补平面字符计为 1 个而非像 UTF-16 code unit 那样计为 2 个。

    Args:
        key: 原始 ``prompt_cache_key``，通常是 session id；``None`` 表示未提供。

    Returns:
        长度不超过上限的 key；``key`` 为 ``None`` 时原样返回 ``None``。
    """
    if key is None:
        return None
    chars = list(key)
    if len(chars) <= OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH:
        return key
    return "".join(chars[:OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH])
