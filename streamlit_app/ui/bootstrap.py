# streamlit_app/ui/bootstrap.py
from __future__ import annotations

import os
import streamlit as st
from Shopping_assistant.nlp.runtime.spacy_runtime import load_spacy

@st.cache_resource(show_spinner=False)
def get_nlp():
    return load_spacy("en_core_web_sm")

@st.cache_resource(show_spinner=False)
def warmup_nlp_stack() -> bool:
    """
    Does:
        Preload spaCy + polarity + lexicon + world alias index + (optional) semantic embeddings cache.
    """
    _ = get_nlp()

    from Shopping_assistant.nlp.parsing.polarity import make_free_polarity_fn
    make_free_polarity_fn()

    from Shopping_assistant.nlp.runtime.lexicon import load_default_lexicon
    lex = load_default_lexicon()
    _ = lex.raw_index

    from Shopping_assistant.nlp.llm.analyze_clauses import build_world_alias_index
    build_world_alias_index(include_xkcd=True)

    # optional: force embeddings cache build
    try:
        if os.environ.get("SA_WARMUP_SEMANTIC", "1").strip() in {"1", "true", "True"}:
            keys = list(lex.raw_index.keys())
            if keys:
                from Shopping_assistant.nlp.runtime.lexicon import (
                    _default_semantic_model,
                    _load_or_build_key_embeddings,
                )
                _load_or_build_key_embeddings(keys, _default_semantic_model())
    except Exception:
        pass

    return True
