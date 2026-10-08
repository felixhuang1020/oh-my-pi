"""扫描项目中残留的纯英文注释与 docstring，用于注释中文化验收。

对每个 Python 文件用 ast 提取所有注释行和 docstring，若某条注释几乎不含
中文字符且包含英文字母，则视为残留并打印。允许白名单：纯路径、纯符号、
分隔线（如 ``# -- state ---``）、机器指令（``# type: ignore``、``# noqa``）等。

Examples:
    >>> python scripts/check_chinese_comments.py [root]
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# 纯分隔线 / 路径 / 标识符 / 机器指令 / TODO 标记等允许保持英文
ALLOW_PATTERNS = [
    re.compile(r"^[\s\-=_~#*:]+$"),  # 分隔线
    re.compile(r"^[\w./\\-]+$"),  # 单个词/路径/标识符
    re.compile(r"^\s*:type\b"),
    re.compile(r"py:func|py:class|py:meth"),  # Sphinx 引用
    # 机器可读指令：类型检查器、linter、覆盖率工具读取，不能本地化
    re.compile(r"^\s*(#\s*)?(type:\s*ignore|noqa|pragma:|ruff:|pyright:|mypy:|isort:|flake8:)"),
    re.compile(r"^#!"),  # 脚本 shebang 行
    re.compile(r"^#\s*.*;\s*$"),  # 注释掉的代码行（以分号结尾）
]

# 注释中允许出现的英文词（不影响判定，仅列出供人工复核时过滤）
ENGLISH_ONLY = re.compile(r"[A-Za-z]")


def has_chinese(text: str) -> bool:
    """判断文本中是否含有中文字符。

    Args:
        text: 待检查的文本片段。

    Returns:
        含至少一个中日韩统一表意文字时返回 ``True``。
    """
    return bool(re.search(r"[一-鿿]", text))


def is_allowed(text: str) -> bool:
    """判断某条注释是否属于允许保留英文的白名单。

    Args:
        text: 原始注释文本（含前导 ``#``）。

    Returns:
        命中分隔线、路径、机器指令等白名单模式时返回 ``True``。
    """
    stripped = text.strip()
    if not stripped:
        return True
    return any(p.match(stripped) for p in ALLOW_PATTERNS)


def check_file(path: Path) -> list[str]:
    """扫描单个 Python 文件中的纯英文注释与 docstring。

    Args:
        path: 待扫描的 Python 文件路径。

    Returns:
        形如 ``"文件:行号: 类型: 内容"`` 的问题列表；文件无法解析时返回单条错误说明。
    """
    findings: list[str] = []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f"{path}: 解析失败 {exc}"]

    # 检查模块/类/函数的 docstring
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc and ENGLISH_ONLY.search(doc) and not has_chinese(doc):
                first = doc.strip().splitlines()[0][:100]
                findings.append(f"{path}:{node.lineno}: docstring: {first}")

    # 检查注释行；用 tokenize 避免把字符串里的 "#" 误判为注释
    import tokenize

    try:
        with tokenize.open(str(path)) as fh:
            for tok in tokenize.generate_tokens(fh.readline):
                if tok.type == tokenize.COMMENT:
                    text = tok.string
                    if has_chinese(text) or is_allowed(text):
                        continue
                    if ENGLISH_ONLY.search(text):
                        findings.append(f"{path}:{tok.start[0]}: comment: {text[:100]}")
    except (tokenize.TokenError, SyntaxError, ValueError) as exc:
        findings.append(f"{path}: tokenize 失败 {exc}")
    return findings


def main() -> int:
    """命令行入口：扫描根目录下全部 Python 文件并打印英文残留。

    Returns:
        存在英文残留时返回 ``1``，全部为中文时返回 ``0``。
    """
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    candidates = [root] if root.is_file() else sorted(root.rglob("*.py"))
    all_findings: list[str] = []
    for path in candidates:
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        all_findings.extend(check_file(path))
    for line in all_findings:
        print(line)
    print(f"\n共 {len(all_findings)} 处残留英文注释/docstring")
    return 1 if all_findings else 0


if __name__ == "__main__":
    sys.exit(main())
