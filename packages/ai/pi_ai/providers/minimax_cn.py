"""MiniMax CN provider 工厂。

移植自 ``packages/ai/src/providers/minimax-cn.ts``。
"""

from __future__ import annotations

from ..api.lazy import lazy_api, lazy_load
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, ProviderImpl, create_provider
from .catalog import chat_model_catalog

__all__ = ["minimax_cn_provider"]


def minimax_cn_provider() -> ProviderImpl:
    """构建 MiniMax CN provider。"""
    return create_provider(
        CreateProviderOptions(
            id="minimax-cn",
            name="MiniMax CN",
            base_url="https://api.minimaxi.com/anthropic",
            auth=ProviderAuth(api_key=env_api_key_auth("MiniMax CN API key", ["MINIMAX_CN_API_KEY"])),
            models=list(chat_model_catalog("minimax-cn").values()),
            api=lazy_api(lazy_load("pi_ai.api.anthropic_messages")),
        )
    )
