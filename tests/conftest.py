# tests/conftest.py
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import importlib.util
import pytest


def _has_sentence_transformers() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _has_sentence_transformers():
        return

    skip = pytest.mark.skip(reason="sentence-transformers not installed (optional NLP dependency)")
    for item in items:
        nodeid = item.nodeid.lower()
        if (
            "test_preference_interpreter_contract" in nodeid
            or "test_constraints_contract" in nodeid
            or "test_bright_ligthness" in nodeid
            or "test_reco_contract" in nodeid
        ):
            item.add_marker(skip)
