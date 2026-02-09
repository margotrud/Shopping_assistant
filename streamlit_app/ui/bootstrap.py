# streamlit_app/ui/bootstrap.py
from __future__ import annotations

import os
from typing import Any, Optional

import streamlit as st

from Shopping_assistant.nlp.runtime.spacy_runtime import load_spacy


@st.cache_resource(show_spinner=False)
def get_nlp():
    return load_spacy("en_core_web_sm")


def _env_bool(key: str, default: str = "0") -> bool:
    v = os.environ.get(key, default)
    return str(v).strip().lower() in {"1", "true", "yes"}


def _warmup_world_alias_index() -> None:
    """
    Does:
        Best-effort warmup of the world alias index WITHOUT importing LLM modules.
    """
    include_xkcd = _env_bool("SA_WARMUP_XKCD", "0")

    # Preferred location (non-LLM). If you refactor later, keep this import working.
    try:
        from Shopping_assistant.nlp.resources.world_alias_index import build_world_alias_index  # type: ignore

        build_world_alias_index(include_xkcd=include_xkcd)
        return
    except Exception:
        pass

    # Backward-compatible fallback (still non-LLM).
    try:
        from Shopping_assistant.nlp.runtime.world_aliases import build_world_alias_index  # type: ignore

        build_world_alias_index(include_xkcd=include_xkcd)
        return
    except Exception:
        return


@st.cache_resource(show_spinner=False)
def warmup_nlp_stack() -> bool:
    """
    Does:
        Preload spaCy + lexicon; optional world-alias warmup (no LLM imports).
        Polarity warmup is OFF by default (can trigger SentenceTransformer cold start).
        Semantic embeddings warmup is OFF by default (too slow for UI startup).
    """
    # Always: spaCy (core parsing)
    _ = get_nlp()

    # Lexicon warmup (moderate)
    lex: Optional[Any] = None
    try:
        from Shopping_assistant.nlp.runtime.lexicon import load_default_lexicon  # type: ignore

        lex = load_default_lexicon()
        _ = getattr(lex, "raw_index", None)
    except Exception:
        lex = None

    # World alias index warmup (keep light; xkcd disabled by default)
    _warmup_world_alias_index()

    # Polarity warmup: OFF by default (can be expensive if backend is semantic).
    # Enable explicitly with SA_WARMUP_POLARITY=1. Backend controlled by SA_POLARITY_BACKEND.
    try:
        do_pol = _env_bool("SA_WARMUP_POLARITY", "0")
        if do_pol:
            from Shopping_assistant.nlp.polarity import make_polarity_fn  # type: ignore

            _ = make_polarity_fn()
    except Exception:
        pass

    # Semantic embeddings warmup (VERY EXPENSIVE): OFF by default.
    # Enable explicitly with SA_WARMUP_SEMANTIC=1 and optionally SA_WARMUP_SEMANTIC_LIMIT=N
    try:
        do_sem = _env_bool("SA_WARMUP_SEMANTIC", "0")
        if do_sem and lex is not None:
            keys = list(getattr(lex, "raw_index", {}).keys())
            if keys:
                limit_raw = os.environ.get("SA_WARMUP_SEMANTIC_LIMIT", "").strip()
                if limit_raw:
                    try:
                        n = int(limit_raw)
                        if n > 0:
                            keys = keys[:n]
                    except Exception:
                        pass

                from Shopping_assistant.nlp.runtime.lexicon import (  # type: ignore
                    _default_semantic_model,
                    _load_or_build_key_embeddings,
                )

                _load_or_build_key_embeddings(keys, _default_semantic_model())
    except Exception:
        pass

    return True
