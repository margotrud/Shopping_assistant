# src/Shopping_assistant/nlp/polarity.py
from __future__ import annotations

"""
Polarity classification for color mentions (LIKE / DISLIKE / None).

Design (portfolio / offline):
- No paid APIs.
- No lexical trigger rules.
- Semantic inference implemented via SentenceTransformer embeddings.

Contract:
    polarity_fn(clause_text, mentions) -> {mention: "LIKE"|"DISLIKE"|None}

This module only:
- normalizes returned labels,
- applies optional clause-level fallback sentiment,
- applies optional structural bias for elliptical fragments.

Important:
- numpy and sentence_transformers are OPTIONAL dependencies and are only required
  when calling make_free_polarity_fn(). The core inference path must remain import-safe.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Protocol

from Shopping_assistant.utils.optional_deps import require

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------


class PolarityLLM(Protocol):
    def __call__(self, clause_text: str, mentions: List[str]) -> Dict[str, Optional[str]]:
        """
        Expected return format:
            {"pink": "LIKE", "red": "DISLIKE", "burgundy": None}
        """
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canon_key(x: str) -> str:
    return x.strip().lower()


def _norm_label(val: object) -> Optional[str]:
    if isinstance(val, str):
        v = val.strip().upper()
        if v in {"LIKE", "DISLIKE"}:
            return v
    return None


def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _l2_normalize_rows(np: Any, X: Any, *, eps: float = 1e-12) -> Any:
    """
    Does:
        L2-normalize rows of a 2D numpy array with numerical stability.
    """
    if getattr(X, "ndim", None) != 2:
        raise RuntimeError(f"Expected a 2D array to normalize, got ndim={getattr(X, 'ndim', None)}")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + float(eps))


# ---------------------------------------------------------------------------
# Core polarity inference (callback + structural rules)
# ---------------------------------------------------------------------------


def infer_polarity_for_mentions(
    clause_text: str,
    mentions: List[str],
    *,
    llm_polarity_fn: PolarityLLM,
    clause_sentiment: Optional[str] = None,
    elliptical_neg: bool = False,
) -> Dict[str, Optional[str]]:
    if not mentions:
        return {}

    if llm_polarity_fn is None:
        raise RuntimeError("infer_polarity_for_mentions requires llm_polarity_fn.")

    try:
        raw = llm_polarity_fn(clause_text, mentions) or {}
    except Exception as e:
        raise RuntimeError(f"llm_polarity_fn failed: {e}") from e

    sent: Optional[str] = None
    if isinstance(clause_sentiment, str):
        s = clause_sentiment.strip().upper()
        if s in {"POS", "NEG"}:
            sent = s

    raw_by_key: Dict[str, object] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(k, str):
                raw_by_key[_canon_key(k)] = v

    out: Dict[str, Optional[str]] = {}
    for m in mentions:
        v_raw = raw_by_key.get(_canon_key(m))
        label = _norm_label(v_raw)

        if label is None and sent is not None:
            label = "LIKE" if sent == "POS" else "DISLIKE"

        if elliptical_neg and label is None:
            label = "DISLIKE"

        out[m] = label

    return out


# ---------------------------------------------------------------------------
# Offline polarity backend (SentenceTransformer)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def make_free_polarity_fn(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    min_sim: float = 0.30,
    min_margin: float = 0.10,
    debug: bool = False,
) -> PolarityLLM:
    """
    Offline polarity:
      - encodes INCLUDE vs EXCLUDE prompts per mention
      - compares to anchors ("include items" / "exclude items")
      - returns None for ambiguous cases based on (min_sim, min_margin)

    Optional deps are imported lazily to keep module import-safe.

    Notes:
      - We DO NOT rely on SentenceTransformer.encode(normalize_embeddings=...)
        for forward-compat across versions; we normalize with numpy explicitly.
    """
    np = require(
        "numpy",
        extra="numpy",
        purpose="Needed for make_free_polarity_fn() to build dense embedding arrays and normalize them.",
    )

    st = require(
        "sentence_transformers",
        extra="sentence-transformers",
        purpose="Needed for make_free_polarity_fn() embedding-based polarity.",
    )
    SentenceTransformer = st.SentenceTransformer  # type: ignore[attr-defined]

    min_sim_f = float(min_sim)
    min_margin_f = float(min_margin)
    if min_sim_f < 0.0:
        min_sim_f = 0.0
    if min_margin_f < 0.0:
        min_margin_f = 0.0

    encoder = SentenceTransformer(model_name)

    anchor_like = "The request asks to include items matching the mention."
    anchor_dislike = "The request asks to exclude items matching the mention."

    anchor_vecs = np.asarray(encoder.encode([anchor_like, anchor_dislike]), dtype=float)
    if getattr(anchor_vecs, "ndim", None) != 2 or anchor_vecs.shape[0] != 2:
        raise RuntimeError(f"Unexpected anchor embedding shape: got_shape={getattr(anchor_vecs, 'shape', None)}")
    anchor_vecs = _l2_normalize_rows(np, anchor_vecs)

    def _fn(clause_text: str, mentions: List[str]) -> Dict[str, Optional[str]]:
        if not mentions:
            return {}

        mentions_u = _dedup_preserve_order(mentions)

        prompts: List[str] = []
        for m in mentions_u:
            prompts.append(
                f"REQUEST: {clause_text}\nMENTION: {m}\nThe request asks to INCLUDE items matching the mention."
            )
            prompts.append(
                f"REQUEST: {clause_text}\nMENTION: {m}\nThe request asks to EXCLUDE items matching the mention."
            )

        vecs = np.asarray(encoder.encode(prompts), dtype=float)
        expected_rows = 2 * len(mentions_u)
        if getattr(vecs, "ndim", None) != 2 or vecs.shape[0] != expected_rows:
            raise RuntimeError(
                "Unexpected prompt embedding shape: "
                f"got_shape={getattr(vecs, 'shape', None)} expected_rows={expected_rows}"
            )
        vecs = _l2_normalize_rows(np, vecs)

        out_u: Dict[str, Optional[str]] = {}
        for i, m in enumerate(mentions_u):
            v_inc = vecs[2 * i]
            v_exc = vecs[2 * i + 1]

            sims_inc = anchor_vecs @ v_inc
            sims_exc = anchor_vecs @ v_exc

            inc_like = float(sims_inc[0])
            inc_dislike = float(sims_inc[1])
            exc_like = float(sims_exc[0])
            exc_dislike = float(sims_exc[1])

            inc_margin = inc_like - inc_dislike
            exc_margin = exc_dislike - exc_like
            score = inc_margin - exc_margin

            max_sim = max(inc_like, inc_dislike, exc_like, exc_dislike)
            if max_sim < min_sim_f or abs(score) < min_margin_f:
                out_u[m] = None
                if debug:
                    log.debug(
                        "[polarity][ambig] mention=%r inc_like=%.3f inc_dislike=%.3f exc_like=%.3f exc_dislike=%.3f "
                        "inc_margin=%.3f exc_margin=%.3f score=%.3f min_sim=%.2f min_margin=%.2f text=%r",
                        m,
                        inc_like,
                        inc_dislike,
                        exc_like,
                        exc_dislike,
                        inc_margin,
                        exc_margin,
                        score,
                        min_sim_f,
                        min_margin_f,
                        clause_text,
                    )
            else:
                out_u[m] = "LIKE" if score > 0 else "DISLIKE"
                if debug:
                    log.debug(
                        "[polarity][decide] mention=%r -> %s inc_like=%.3f inc_dislike=%.3f exc_like=%.3f exc_dislike=%.3f "
                        "inc_margin=%.3f exc_margin=%.3f score=%.3f text=%r",
                        m,
                        out_u[m],
                        inc_like,
                        inc_dislike,
                        exc_like,
                        exc_dislike,
                        inc_margin,
                        exc_margin,
                        score,
                        clause_text,
                    )

        # Preserve original mentions list (including duplicates) in output keys
        return {m: out_u.get(m) for m in mentions}

    return _fn


__all__ = [
    "PolarityLLM",
    "infer_polarity_for_mentions",
    "make_free_polarity_fn",
]
