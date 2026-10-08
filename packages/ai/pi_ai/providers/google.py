"""Google provider 工厂。

移植自 ``packages/ai/src/providers/google.ts``。
"""

from __future__ import annotations

from ..api.lazy import lazy_api, lazy_load
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..models import CreateProviderOptions, ProviderImpl, create_provider
from .catalog import chat_model_catalog

__all__ = ["google_provider"]


def google_provider() -> ProviderImpl:
    """构建 Google provider。"""
    return create_provider(
        CreateProviderOptions(
            id="google",
            name="Google",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            auth=ProviderAuth(api_key=env_api_key_auth("Gemini API key", ["GEMINI_API_KEY"])),
            models=list(chat_model_catalog("google").values()),
            api=lazy_api(lazy_load("pi_ai.api.google_generative_ai")),
        )
    )
