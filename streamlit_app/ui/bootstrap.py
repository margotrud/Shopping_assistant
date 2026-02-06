# streamlit_app/ui/bootstrap.py
from __future__ import annotations

import os
from typing import Any, Optional

import streamlit as st

from Shopping_assistant.nlp.runtime.spacy_runtime import load_spacy


@st.cache_resource(show_spinner=False)
def get_nlp():
    return load_spacy("en_core_web_sm")


def _warmup_world_alias_index() -> None:
    """
    Does:
        Best-effort warmup of the world alias index WITHOUT importing LLM modules.
    """
    # Preferred location (non-LLM). If you refactor later, keep this import working.
    try:
        from Shopping_assistant.nlp.resources.world_alias_index import build_world_alias_index  # type: ignore

        build_world_alias_index(include_xkcd=True)
        return
    except Exception:
        pass

    # Backward-compatible: if the function still lives elsewhere, try it but NEVER crash warmup.
    # NOTE: We explicitly avoid importing Shopping_assistant.nlp.llm.analyze_clauses here.
    try:
        from Shopping_assistant.nlp.runtime.world_aliases import build_world_alias_index  # type: ignore

        build_world_alias_index(include_xkcd=True)
        return
    except Exception:
        return


@st.cache_resource(show_spinner=False)
def warmup_nlp_stack() -> bool:
    """
    Does:
        Preload spaCy + polarity + lexicon + (optional) semantic embeddings cache.
        World-alias warmup is best-effort and MUST NOT import LLM modules.
    """
    _ = get_nlp()

    # polarity fn warmup
    try:
        from Shopping_assistant.nlp.parsing.polarity import make_free_polarity_fn  # type: ignore

        make_free_polarity_fn()
    except Exception:
        # keep warmup non-blocking
        pass

    # lexicon warmup
    lex: Optional[Any] = None
    try:
        from Shopping_assistant.nlp.runtime.lexicon import load_default_lexicon  # type: ignore

        lex = load_default_lexicon()
        _ = getattr(lex, "raw_index", None)
    except Exception:
        lex = None

    # world alias index warmup (non-blocking, non-LLM)
    _warmup_world_alias_index()

    # optional: force embeddings cache build (non-blocking)
    try:
        if os.environ.get("SA_WARMUP_SEMANTIC", "1").strip() in {"1", "true", "True"} and lex is not None:
            keys = list(getattr(lex, "raw_index", {}).keys())
            if keys:
                from Shopping_assistant.nlp.runtime.lexicon import (  # type: ignore
                    _default_semantic_model,
                    _load_or_build_key_embeddings,
                )

                _load_or_build_key_embeddings(keys, _default_semantic_model())
    except Exception:
        pass

    return True
