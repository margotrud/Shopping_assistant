# src/Shopping_assistant/nlp/runtime/spacy_runtime.py
from __future__ import annotations

from functools import lru_cache
from typing import Any

from Shopping_assistant.utils.optional_deps import require


@lru_cache(maxsize=2)
def load_spacy(model: str):
    """
    Does:
        Load and cache a spaCy Language pipeline for the given model name.
    """
    spacy = require(
        "spacy",
        extra="spacy",
        purpose="Needed for NLP parsing (single cached spaCy runtime).",
    )
    return spacy.load(model)
