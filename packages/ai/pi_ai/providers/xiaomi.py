"""Xiaomi provider 工厂。

移植自 ``packages/ai/src/providers/xiaomi.ts``。
"""

from __future__ import annotations

from ..api.lazy import lazy_api, lazy_load
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, ProviderImpl, create_provider
from .catalog import chat_model_catalog

__all__ = ["xiaomi_provider"]


def xiaomi_provider() -> ProviderImpl:
    """构建 Xiaomi provider。"""
    return create_provider(
        CreateProviderOptions(
            id="xiaomi",
            name="Xiaomi",
            base_url="https://api.xiaomimimo.com/v1",
            auth=ProviderAuth(api_key=env_api_key_auth("Xiaomi API key", ["XIAOMI_API_KEY"])),
            models=list(chat_model_catalog("xiaomi").values()),
            api=lazy_api(lazy_load("pi_ai.api.openai_responses")),
        )
    )
