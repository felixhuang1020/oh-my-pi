"""运行时 model 类型收窄助手。"""

from __future__ import annotations

from typing import Any

from ..types import ModelType
from .models_error import ModelsError

__all__ = [
    "assert_chat_model",
    "get_model_type",
    "is_model_type",
]


def get_model_type(model: Any) -> str:
    """获取 model 的类型；没有 ``type`` 属性的 model 视为 chat model.

    本项目只支持 chat 模型：目录加载时会丢弃其他类型，因此这里返回非
    ``"chat"`` 只可能来自手工构造的 model，调用方应将其视为不受支持。

    Args:
        model: 任意 model 对象。

    Returns:
        类型字符串；缺少 ``type`` 属性时返回 ``"chat"``。
    """
    return getattr(model, "type", None) or "chat"


def is_model_type(model: Any, type: ModelType) -> bool:
    """运行时判断 model 是否为给定类型，兼容没有 ``type`` 的旧式 chat model.

    Args:
        model: 任意 model 对象。
        type: 期望的类型；类型面上只允许 ``"chat"``。

    Returns:
        类型一致时为 ``True``，否则为 ``False``。
    """
    return get_model_type(model) == type


def assert_chat_model(model: Any) -> None:
    """若 ``model`` 不是 chat model 则抛出 :class:`ModelsError`.

    Args:
        model: 待校验的 model 对象。

    Raises:
        ModelsError: 当 ``model`` 的类型不是 ``"chat"`` 时。
    """
    if not is_model_type(model, "chat"):
        raise ModelsError("provider", f"Model {model.provider}/{model.id} is not a chat model")
