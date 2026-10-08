#!/usr/bin/env python3
"""报告 TypeScript → Python 的移植覆盖率。

遍历上游 TypeScript 源码与 Python 包，把每个源文件与承载它的 Python 模块配对，
并打印尚未移植的条目。

Examples:
    >>> uv run python scripts/porting_status.py
    >>> uv run python scripts/porting_status.py --ts-root /path/to/WebStromProjects/pi/packages
"""

from __future__ import annotations

import argparse
import re
import sys
from fnmatch import fnmatch
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TS_ROOT = Path("/Users/huangzd00/Projects/WebStromProjects/pi/packages")

#: 有意合并到其他 Python 模块中的 TypeScript 文件。
FOLDED: dict[str, str] = {
    "ai/src/utils/abort-signals.ts": "pi_ai/utils/abort.py (combine_abort_signals)",
    "ai/src/index.ts": "pi_ai/__init__.py",
    "agent/src/index.ts": "pi_agent/__init__.py",
    "ai/src/providers/*.models.ts": "pi_ai/providers/catalog.py (data-driven)",
    "ai/src/api/*.lazy.ts": "lazy_api(lazy_load(...)) call sites in the provider factories",
}

#: 本 fork 有意删除的上游模块（对应已删除的 Python 模块）。这些不是移植缺口。
REMOVED_BY_FORK: tuple[str, ...] = (
    # 被裁剪掉的 wire protocol
    "ai/src/api/openai-completions.ts",
    "ai/src/api/azure-openai-responses.ts",
    "ai/src/api/bedrock-converse-stream.ts",
    "ai/src/api/cloudflare.ts",
    "ai/src/api/cloudflare-ai-binding.ts",
    "ai/src/api/cloudflare-workers-ai-system-one.ts",
    "ai/src/api/google-vertex.ts",
    "ai/src/api/llama-cpp-classify.ts",
    "ai/src/api/mistral-conversations.ts",
    "ai/src/api/openai-codex-responses.ts",
    "ai/src/api/openrouter-images.ts",
    "ai/src/api/system-one-shared.ts",
    "ai/src/api/typesafe-system-one.ts",
    # 图像生成与分类
    "ai/src/image-models.ts",
    "ai/src/images.ts",
    "ai/src/images-api-registry.ts",
    "ai/src/providers/images/*.ts",
    # OAuth
    "ai/src/oauth.ts",
    "ai/src/bun-oauth.ts",
    "ai/src/utils/oauth-page.ts",
    "ai/src/compat/extension-oauth-types.ts",
    "ai/src/auth/oauth/*.ts",
    # 被裁剪掉的 provider 工厂
    "ai/src/providers/amazon-bedrock.ts",
    "ai/src/providers/ant-ling.ts",
    "ai/src/providers/azure-openai-responses.ts",
    "ai/src/providers/baseten.ts",
    "ai/src/providers/cerebras.ts",
    "ai/src/providers/cloudflare-ai-gateway.ts",
    "ai/src/providers/cloudflare-auth.ts",
    "ai/src/providers/cloudflare-stream.ts",
    "ai/src/providers/cloudflare-workers-ai.ts",
    "ai/src/providers/fireworks.ts",
    "ai/src/providers/github-copilot.ts",
    "ai/src/providers/google-vertex.ts",
    "ai/src/providers/groq.ts",
    "ai/src/providers/huggingface.ts",
    "ai/src/providers/kimi-coding.ts",
    "ai/src/providers/meta.ts",
    "ai/src/providers/mistral.ts",
    "ai/src/providers/nvidia.ts",
    "ai/src/providers/openai-codex.ts",
    "ai/src/providers/opencode.ts",
    "ai/src/providers/opencode-go.ts",
    "ai/src/providers/opencode-headers.ts",
    "ai/src/providers/openrouter.ts",
    "ai/src/providers/qwen-token-plan.ts",
    "ai/src/providers/qwen-token-plan-cn.ts",
    "ai/src/providers/qwen-token-plan-individual.ts",
    "ai/src/providers/radius.ts",
    "ai/src/providers/radius-config.ts",
    "ai/src/providers/together.ts",
    "ai/src/providers/typesafe.ts",
    "ai/src/providers/vercel-ai-gateway.ts",
    "ai/src/providers/xai.ts",
    "ai/src/providers/xiaomi-token-plan-ams.ts",
    "ai/src/providers/xiaomi-token-plan-cn.ts",
    "ai/src/providers/xiaomi-token-plan-sgp.ts",
    "ai/src/providers/zai.ts",
    "ai/src/providers/zai-coding-cn.ts",
    # 其他被裁剪模块
    "ai/src/legacy-api-aliases.ts",
)

#: 生成产物、纯类型声明或数据文件，跳过不计。
SKIPPED_SUFFIXES = (".d.ts",)
SKIPPED_NAMES = {"models.generated.ts", "data-json.d.ts"}


def _snake(name: str) -> str:
    """把 TypeScript 文件名转换为 Python 的 snake_case 形式。

    Args:
        name: 源文件名（不含扩展名），可能含 ``-`` 或空格。

    Returns:
        把 ``-`` 与空白替换为 ``_`` 后的名字。
    """
    return re.sub(r"[-\s]", "_", name)


def expected_module(rel: Path, package: str) -> Path | None:
    """返回应当承载 ``rel`` 的 Python 模块路径。

    Args:
        rel: 相对于上游 ``src`` 目录的 TypeScript 文件路径。
        package: 所属包名，取 ``"ai"`` 或 ``"agent"``。

    Returns:
        对应的 Python 模块路径；无需独立模块（如惰性包装层、生成的模型垫片）时返回 ``None``。
    """
    root = REPO / "packages" / package / ("pi_ai" if package == "ai" else "pi_agent")
    stem = _snake(rel.stem)
    if rel.stem.endswith(".lazy"):
        return None  # lazy 包装层归并到 lazy_load() 调用处
    if rel.stem.endswith(".models"):
        return None  # 生成的模型垫片归并到 providers/catalog.py
    parent = rel.parent
    if package == "ai" and parent.as_posix() == "providers/images":
        return REPO / "packages/ai/pi_ai/providers/images" / f"{stem}.py"
    return root / parent / f"{stem}.py"


def main() -> int:
    """命令行入口：统计移植覆盖率并打印报告。

    Returns:
        存在未移植文件时返回 ``1``，全部覆盖返回 ``0``，上游目录缺失返回 ``2``。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ts-root", type=Path, default=DEFAULT_TS_ROOT)
    args = parser.parse_args()

    if not args.ts_root.is_dir():
        print(f"TypeScript sources not found at {args.ts_root}", file=sys.stderr)
        return 2

    present: list[str] = []
    missing: list[str] = []
    folded: list[str] = []
    removed: list[str] = []

    for package in ("ai", "agent"):
        source_root = args.ts_root / package / "src"
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*.ts")):
            rel = path.relative_to(source_root)
            if rel.name.endswith(SKIPPED_SUFFIXES) or rel.name in SKIPPED_NAMES:
                continue
            key = f"{package}/src/{rel.as_posix()}"
            if key in FOLDED:
                folded.append(f"{key} -> {FOLDED[key]}")
                continue
            if rel.parts[:1] == ("providers",) and rel.parts[1:2] == ("data",):
                continue
            if rel.stem.endswith((".lazy", ".models")):
                folded.append(f"{key} -> {FOLDED['ai/src/api/*.lazy.ts']}")
                continue
            if any(fnmatch(key, pattern) for pattern in REMOVED_BY_FORK):
                removed.append(key)
                continue
            target = expected_module(rel, package)
            (present if target is not None and target.exists() else missing).append(key)

    total = len(present) + len(missing) + len(folded) + len(removed)
    print(f"TypeScript source files considered : {total}")
    print(f"carried by a Python module        : {len(present)}")
    print(f"intentionally folded              : {len(folded)}")
    print(f"removed by this fork              : {len(removed)}")
    print(f"NOT YET PORTED                    : {len(missing)}")
    if missing:
        print()
        for entry in missing:
            print(f"  missing  {entry}")
    if folded:
        print()
        for entry in folded:
            print(f"  folded   {entry}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
