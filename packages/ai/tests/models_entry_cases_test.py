"""移植自 ``packages/ai/test/models-entry.test.ts``（"轻量 models 入口"）。

上游通过 Node loader 钩子在轻量 models 入口加载任何重依赖（TypeBox、各 provider SDK、
生成的目录数据、``models.generated``）时直接抛错，然后执行一次 faux 补全。
Python 版在子解释器中运行同样的脚本化断言：若入口拉入了
``pi_ai.providers.catalog``、``pi_ai.models_generated`` 或任何具体的 ``pi_ai.api.*``
适配器（loader 本身 ``pi_ai.api.lazy`` 则允许），测试即失败。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

PROBE = "\n".join(
    [
        "import asyncio, sys",
        "from pi_ai.models import create_models, create_provider",
        "from pi_ai.providers.faux import faux_assistant_message, faux_provider",
        "from pi_ai.types import Context",
        "from pi_ai.utils.serde import to_json",
        "",
        "forbidden = [",
        "    module",
        "    for module in sys.modules",
        "    if module in ('pi_ai.providers.catalog', 'pi_ai.models_generated')",
        "    or (module.startswith('pi_ai.api.') and module != 'pi_ai.api.lazy')",
        "]",
        "assert not forbidden, 'Heavy dependency loaded by models entry: ' + repr(forbidden)",
        "",
        "assert callable(create_provider)",
        "models = create_models()",
        "assert models.get_models() == [], models.get_models()",
        "assert models.get_providers() == [], models.get_providers()",
        "",
        "async def main():",
        "    faux = faux_provider()",
        "    models.set_provider(faux.provider)",
        "    faux.set_responses([faux_assistant_message('OK')])",
        "    response = await models.complete_simple(faux.get_model(), Context(messages=[]))",
        "    assert response.stop_reason == 'stop', response.stop_reason",
        "    assert to_json(response.content) == [{'type': 'text', 'text': 'OK'}], to_json(response.content)",
        "",
        "asyncio.run(main())",
    ]
)


def test_runs_a_faux_completion_without_typebox_catalogs_or_sdks_through_pi_ai_models():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PACKAGE_ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(PACKAGE_ROOT)]
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", PROBE],
            cwd=str(PACKAGE_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except subprocess.TimeoutExpired as error:  # 上游断言 `result.error` 为 undefined
        pytest.fail(f"probe timed out: {error}")

    # 上游还对包引用说明符做了参数化
    # （`@earendil-works/pi-ai/models`）；移植版只有一个 `pi_ai.models` 入口。
    assert result.returncode == 0, result.stderr
