"""``pi-ai`` 包的简易 api-key 登录 CLI。

零第三方依赖（仅标准库），import 时无副作用：
入口点置于 ``if __name__ == "__main__":`` 之下。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .auth.types import (
    ApiKeyCredential,
    AuthPrompt,
    ProviderAuthInteraction,
    ProviderAuthInteractionImpl,
)
from .providers.all import builtin_providers
from .utils.abort import AbortSignal

__all__ = ["AUTH_FILE", "PROVIDERS", "answer_prompt", "load_auth", "login", "main", "save_auth"]

AUTH_FILE = "auth.json"

#: 暴露 api-key 登录流程的 provider。
PROVIDERS = [provider for provider in builtin_providers() if provider.auth.api_key is not None]


async def _prompt(question: str) -> str:
    """从 stdin 读取一行，不阻塞事件循环。

    Args:
        question: 显示给用户的提示文本。

    Returns:
        用户输入的整行内容（不含末尾换行）。
    """
    return await asyncio.to_thread(input, question)


def load_auth() -> dict[str, ApiKeyCredential]:
    """读取本地 ``auth.json`` 凭据文件，文件缺失或损坏时返回空字典。

    Returns:
        以 provider id 为键的凭据映射；读取或解析失败时为空字典。
    """
    path = Path(AUTH_FILE)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_auth(auth: dict[str, ApiKeyCredential]) -> None:
    """将凭据写入本地 ``auth.json`` 文件。

    Args:
        auth: 以 provider id 为键的凭据映射。
    """
    Path(AUTH_FILE).write_text(json.dumps(auth, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    """把 dataclass 等 JSON 不认识的值交给 :func:`pi_ai.utils.serde.to_json` 处理。

    Args:
        value: :mod:`json` 无法直接序列化的值。

    Returns:
        可被 :mod:`json` 继续序列化的表示。
    """
    from .utils.serde import to_json

    return to_json(value)


async def answer_prompt(auth_prompt: AuthPrompt) -> str:
    """渲染单个 provider 认证提示，并返回输入的值。

    Args:
        auth_prompt: 待渲染的认证提示。

    Returns:
        用户输入的文本。
    """
    placeholder = f" ({auth_prompt.placeholder})" if getattr(auth_prompt, "placeholder", None) else ""
    return await _prompt(f"{auth_prompt.message}{placeholder}: ")


class _CliInteraction:
    """开发用 CLI 的控制台版 :class:`~pi_ai.auth.types.AuthInteraction`。"""

    signal = AbortSignal()

    async def prompt(self, prompt: AuthPrompt) -> str:
        """把 provider 认证提示转交给控制台实现。

        Args:
            prompt: 待渲染的认证提示。

        Returns:
            用户输入的值。
        """
        return await answer_prompt(prompt)


async def login(provider_id: str) -> None:
    """执行单个 provider 的 api-key 登录流程并保存凭据。

    Args:
        provider_id: 目标 provider id。

    Raises:
        RuntimeError: 该 provider id 不存在或不支持 api-key 登录。
    """
    provider = next((entry for entry in PROVIDERS if entry.id == provider_id), None)
    if provider is None:
        raise RuntimeError(f"Unknown provider: {provider_id}")
    interaction: ProviderAuthInteraction = ProviderAuthInteractionImpl(_CliInteraction())
    credential = await provider.auth.api_key.login(interaction)
    auth = load_auth()
    auth[provider_id] = credential
    save_auth(auth)
    print(f"\nCredentials saved to {AUTH_FILE}")


async def main() -> None:
    """分发 ``login``/``list``/``help`` 子命令。"""
    args = sys.argv[1:]
    command = args[0] if args else None
    if not command or command in ("help", "--help", "-h"):
        provider_list = "\n".join(f"  {provider.id:<20} {provider.name}" for provider in PROVIDERS)
        print(
            "Usage: python -m pi_ai.cli <command> [provider]\n\n"
            "Commands:\n"
            "  login [provider]  Store an api key for a provider\n"
            "  list              List available providers\n\n"
            f"Providers:\n{provider_list}"
        )
        return
    if command == "list":
        for provider in PROVIDERS:
            print(f"{provider.id:<20} {provider.name}")
        return
    if command == "login":
        provider_id = args[1] if len(args) > 1 else None
        if not provider_id:
            for index, provider in enumerate(PROVIDERS):
                print(f"  {index + 1}. {provider.name}")
            answer = await _prompt(f"Enter number (1-{len(PROVIDERS)}): ")
            try:
                index = int(answer.strip()) - 1
            except ValueError:
                index = -1
            provider_id = PROVIDERS[index].id if 0 <= index < len(PROVIDERS) else None
        if not provider_id or not any(provider.id == provider_id for provider in PROVIDERS):
            raise RuntimeError(f"Unknown provider: {provider_id or ''}")
        await login(provider_id)
        return
    raise RuntimeError(f"Unknown command: {command}")


def _run_cli() -> None:
    """同步入口：运行 :func:`main`，并统一处理顶层错误。"""
    try:
        asyncio.run(main())
    except Exception as error:  # noqa: BLE001 - CLI 顶层错误边界
        print("Error:", str(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    _run_cli()
