"""Top-level package exports for llm_harness."""

from importlib import import_module
from types import ModuleType

__all__ = ["agents", "clients", "stats", "tools", "utils"]


def __getattr__(name: str) -> ModuleType:
    """Lazily import top-level subpackages on first access."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
