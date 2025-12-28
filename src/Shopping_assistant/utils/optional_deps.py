# src/Shopping_assistant/utils/optional_deps.py
from __future__ import annotations

import importlib
from typing import Any


def require(module_name: str, *, extra: str, purpose: str) -> Any:
    """
    Does:
        Import an optional dependency or raise a clear error with install hint.
    """
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise RuntimeError(
            f"Optional dependency missing: {module_name}\n"
            f"Install: pip install {extra}\n"
            f"Purpose: {purpose}"
        ) from e
