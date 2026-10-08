"""Moonshot AI provider 工厂。

移植自 ``packages/ai/src/providers/moonshotai.ts``。
"""

from __future__ import annotations

from ..api.lazy import lazy_api, lazy_load
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, ProviderImpl, create_provider
from .catalog import chat_model_catalog

__all__ = ["moonshotai_provider"]


def moonshotai_provider() -> ProviderImpl:
    """构建 Moonshot AI provider。"""
    return create_provider(
        CreateProviderOptions(
            id="moonshotai",
            name="Moonshot AI",
            base_url="https://api.moonshot.ai/v1",
            auth=ProviderAuth(api_key=env_api_key_auth("Moonshot AI API key", ["MOONSHOT_API_KEY"])),
            models=list(chat_model_catalog("moonshotai").values()),
            api=lazy_api(lazy_load("pi_ai.api.openai_responses")),
        )
    )
