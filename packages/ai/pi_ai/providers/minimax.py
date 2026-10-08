"""MiniMax provider 工厂。

移植自 ``packages/ai/src/providers/minimax.ts``。
"""

from __future__ import annotations

from ..api.lazy import lazy_api, lazy_load
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, ProviderImpl, create_provider
from .catalog import chat_model_catalog

__all__ = ["minimax_provider"]


def minimax_provider() -> ProviderImpl:
    """构建 MiniMax provider。"""
    return create_provider(
        CreateProviderOptions(
            id="minimax",
            name="MiniMax",
            base_url="https://api.minimax.io/anthropic",
            auth=ProviderAuth(api_key=env_api_key_auth("MiniMax API key", ["MINIMAX_API_KEY"])),
            models=list(chat_model_catalog("minimax").values()),
            api=lazy_api(lazy_load("pi_ai.api.anthropic_messages")),
        )
    )
