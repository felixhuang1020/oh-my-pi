"""OpenAI provider 工厂。

移植自 ``packages/ai/src/providers/openai.ts``。
"""

from __future__ import annotations

from ..api.lazy import lazy_api, lazy_load
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, ProviderImpl, create_provider
from .catalog import chat_model_catalog

__all__ = ["openai_provider"]


def openai_provider() -> ProviderImpl:
    """构建 OpenAI provider。"""
    return create_provider(
        CreateProviderOptions(
            id="openai",
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            auth=ProviderAuth(api_key=env_api_key_auth("OpenAI API key", ["OPENAI_API_KEY"])),
            models=list(chat_model_catalog("openai").values()),
            api=lazy_api(lazy_load("pi_ai.api.openai_responses")),
        )
    )
