# src/Shopping_assistant/nlp/runtime/spacy_runtime.py
from __future__ import annotations

from functools import lru_cache

from Shopping_assistant.utils.optional_deps import require


@lru_cache(maxsize=2)
def load_spacy(model: str):
    """
    Does:
        Load and cache a spaCy Language pipeline for the given model name.
        Guaranteed single load per model name across the process.
    """
    spacy = require(
        "spacy",
        extra="spacy",
        purpose="NLP parsing (cached spaCy runtime).",
    )

    # CRITICAL: spacy.load MUST NOT be called outside this function
    return spacy.load(model)
