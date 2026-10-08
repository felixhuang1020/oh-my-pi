"""DeepSeek provider 工厂。

移植自 ``packages/ai/src/providers/deepseek.ts``。
"""

from __future__ import annotations

from ..api.lazy import lazy_api, lazy_load
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, ProviderImpl, create_provider
from .catalog import chat_model_catalog

__all__ = ["deepseek_provider"]


def deepseek_provider() -> ProviderImpl:
    """构建 DeepSeek provider。"""
    return create_provider(
        CreateProviderOptions(
            id="deepseek",
            name="DeepSeek",
            base_url="https://api.deepseek.com",
            auth=ProviderAuth(api_key=env_api_key_auth("DeepSeek API key", ["DEEPSEEK_API_KEY"])),
            models=list(chat_model_catalog("deepseek").values()),
            api=lazy_api(lazy_load("pi_ai.api.openai_responses")),
        )
    )
