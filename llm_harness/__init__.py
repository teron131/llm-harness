"""LangChain Playground - A collection of LangChain utilities and tools"""

from __future__ import annotations

import ast
from functools import cache
import importlib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_PACKAGE_DIR = Path(__file__).resolve().parent
_EXPORT_SOURCES: list[tuple[str, Path]] = [
    ("llm_harness.clients", _PACKAGE_DIR / "clients" / "__init__.py"),
    ("llm_harness.fs_tools", _PACKAGE_DIR / "fs_tools.py"),
    ("llm_harness.image_utils", _PACKAGE_DIR / "image_utils.py"),
    ("llm_harness.text_utils", _PACKAGE_DIR / "text_utils.py"),
    ("llm_harness.tools", _PACKAGE_DIR / "tools" / "__init__.py"),
]


@cache
def _parse_dunder_all(file_path: str) -> list[str]:
    try:
        source = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            continue
        value = node.value
        if not isinstance(value, (ast.List, ast.Tuple)):
            return []
        names: list[str] = []
        for elt in value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                names.append(elt.value)
        return names

    return []


def _build_lazy_exports() -> dict[str, tuple[str, str]]:
    exports: dict[str, tuple[str, str]] = {}
    for module_name, file_path in _EXPORT_SOURCES:
        for name in _parse_dunder_all(str(file_path)):
            exports.setdefault(name, (module_name, name))
    return exports


_LAZY_EXPORTS = _build_lazy_exports()


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr = target
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_EXPORTS.keys()))


__all__ = sorted(_LAZY_EXPORTS.keys())
