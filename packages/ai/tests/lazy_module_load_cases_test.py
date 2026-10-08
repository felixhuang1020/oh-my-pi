"""移植 ``packages/ai/test/lazy-module-load.test.ts``（"lazy provider module loading"）。

上游启动一个带 module-resolve 钩子的 Node 子进程，记录进程加载的每个 SDK 标识符
（``@anthropic-ai/sdk``、``openai`` 等）。Python 移植版没有 SDK 模块；忠实的对应物
是启动一个子解释器导入相同的入口点，再检查 ``sys.modules`` 中是否出现具体的
adapter 模块（这正是惰性包装要延迟加载的对象）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

#: 具体的 provider adapter：对应上游的 SDK 标识符清单。
ADAPTER_MODULES = sorted(
    {
        "pi_ai.api.anthropic_messages",
        "pi_ai.api.google_generative_ai",
        "pi_ai.api.openai_responses",
        "pi_ai.api.pi_messages",
    }
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _run_probe(action: str) -> dict:
    """先导入根 barrel，再在子解释器中执行 ``action``。

    对应上游的 ``runProbe``：探针总是先导入包入口点，然后执行 action，
    最后报告加载了哪些 adapter 模块。
    """
    script = "\n".join(
        [
            "import asyncio, json, sys",
            "import pi_ai",
            f"ADAPTER_MODULES = set({ADAPTER_MODULES!r})",
            action,
            "loaded = sorted(module for module in sys.modules if module in ADAPTER_MODULES)",
            "print(json.dumps({'loadedSpecifiers': loaded}))",
        ]
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PACKAGE_ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(PACKAGE_ROOT)]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(PACKAGE_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Probe failed (exit {result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"Probe produced no output\nSTDERR:\n{result.stderr}")
    return json.loads(lines[-1])


def test_does_not_load_provider_sdks_when_importing_the_root_barrel():
    result = _run_probe("")
    assert result["loadedSpecifiers"] == []


def test_does_not_load_provider_sdks_when_building_all_builtin_providers():
    result = _run_probe(
        "\n".join(
            [
                "from pi_ai.providers.all import builtin_models",
                "models = builtin_models()",
                "models.get_models()",
            ]
        )
    )
    assert result["loadedSpecifiers"] == []


def test_does_not_load_provider_sdks_when_importing_the_compat_entrypoint():
    result = _run_probe("import pi_ai.compat")
    assert result["loadedSpecifiers"] == []


def test_loads_only_the_anthropic_sdk_when_streaming_through_the_lazy_api_wrapper():
    result = _run_probe(
        "\n".join(
            [
                "from pi_ai.providers.anthropic import anthropic_provider",
                "from pi_ai.types import Context, UserMessage",
                "from pi_ai.utils.transcript import normalize_context",
                "",
                "async def main():",
                "    provider = anthropic_provider()",
                "    model = provider.get_models()[0]",
                "    context = normalize_context(Context(messages=[UserMessage(content='hi')]))",
                "    await provider.stream_simple(model, context, None).result()",
                "",
                "asyncio.run(main())",
            ]
        )
    )
    assert result["loadedSpecifiers"] == ["pi_ai.api.anthropic_messages"]


def test_loads_only_the_anthropic_sdk_when_dispatching_through_stream_simple():
    result = _run_probe(
        "\n".join(
            [
                "import pi_ai.compat as compat",
                "from pi_ai.types import Context, UserMessage",
                "",
                "async def main():",
                "    model = compat.get_model('anthropic', 'claude-sonnet-4-6')",
                "    await compat.stream_simple(model, Context(messages=[UserMessage(content='hi')])).result()",
                "",
                "asyncio.run(main())",
            ]
        )
    )
    assert result["loadedSpecifiers"] == ["pi_ai.api.anthropic_messages"]
