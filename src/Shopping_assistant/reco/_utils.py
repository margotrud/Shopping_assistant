# src/Shopping_assistant/reco/_utils.py
from __future__ import annotations


def _get(m, key: str, default=None):
    try:
        if isinstance(m, dict):
            return m.get(key, default)
        return getattr(m, key, default)
    except Exception:
        return default


def _get_enum_value(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            return x
        return getattr(x, "value", None)
    except Exception:
        return None
