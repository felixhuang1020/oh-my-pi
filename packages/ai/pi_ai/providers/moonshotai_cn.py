"""Moonshot AI CN provider 工厂。

移植自 ``packages/ai/src/providers/moonshotai-cn.ts``。
"""

from __future__ import annotations

from ..api.lazy import lazy_api, lazy_load
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, ProviderImpl, create_provider
from .catalog import chat_model_catalog

__all__ = ["moonshotai_cn_provider"]


def moonshotai_cn_provider() -> ProviderImpl:
    """构建 Moonshot AI CN provider。"""
    return create_provider(
        CreateProviderOptions(
            id="moonshotai-cn",
            name="Moonshot AI CN",
            base_url="https://api.moonshot.cn/v1",
            auth=ProviderAuth(api_key=env_api_key_auth("Moonshot AI API key", ["MOONSHOT_API_KEY"])),
            models=list(chat_model_catalog("moonshotai-cn").values()),
            api=lazy_api(lazy_load("pi_ai.api.openai_responses")),
        )
    )
