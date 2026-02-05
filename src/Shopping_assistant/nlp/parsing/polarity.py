# src/Shopping_assistant/nlp/polarity.py
from __future__ import annotations

"""
Polarity classification for mentions (LIKE / DISLIKE / UNKNOWN).

Design (portfolio / offline):
- No paid APIs.
- No lexical trigger rules.
- Semantic inference implemented via SentenceTransformer embeddings.

Contracts:

1) Polarity backend (callback):
    polarity_fn(clause_text, mentions) -> {mention: "LIKE"|"DISLIKE"|None}

2) This module:
- normalizes returned labels,
- applies optional clause-level fallback sentiment,
- applies optional structural bias for elliptical fragments,
- provides a deterministic clause-level polarity decision,
- provides enum-typed outputs (Shopping_assistant.nlp.schema.Polarity).

Important:
- numpy and sentence_transformers are OPTIONAL dependencies and are only required
  when calling make_free_polarity_fn(). The core inference path must remain import-safe.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Protocol

from Shopping_assistant.nlp.schema import Polarity
from Shopping_assistant.utils.optional_deps import require

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------


class PolarityLLM(Protocol):
    """Does: optional LLM-backed polarity helper for ambiguous clauses/mentions.
    Provides: a callable interface used by polarity inference when enabled.
    """
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


def _label_to_enum(label: Optional[str]) -> Polarity:
    if label == "LIKE":
        return Polarity.LIKE
    if label == "DISLIKE":
        return Polarity.DISLIKE
    return Polarity.UNKNOWN


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


def _load_sentence_transformer_safe(*, SentenceTransformer: Any, model_name: str, torch: Any) -> Any:
    device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"

    # 1) Chemin robuste: construire les modules ST explicitement
    try:
        from sentence_transformers import models  # type: ignore

        word = models.Transformer(
            model_name,
            model_args={
                "low_cpu_mem_usage": False,
                "device_map": None,
            },
        )
        pooling = models.Pooling(
            word.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
        model = SentenceTransformer(modules=[word, pooling], device=device)
    except Exception:
        # 2) Fallback: constructeur classique
        try:
            model = SentenceTransformer(
                model_name,
                device=device,
                model_kwargs={"low_cpu_mem_usage": False, "device_map": None},
            )
        except TypeError:
            model = SentenceTransformer(model_name, device=device)

    try:
        model.eval()
    except Exception:
        pass

    return model

# ---------------------------------------------------------------------------
# Clause-level polarity decision
# ---------------------------------------------------------------------------


def decide_clause_polarity(
    mention_labels: Dict[str, Optional[str]],
    *,
    clause_sentiment: Optional[str] = None,
    elliptical_neg: bool = False,
) -> Polarity:
    """
    Does:
        Decide a clause-level polarity from mention-level labels + optional clause sentiment,
        with a deterministic fallback for elliptical negative fragments.
    """
    if mention_labels:
        vals = [_label_to_enum(v) for v in mention_labels.values() if v is not None]
        has_like = any(v == Polarity.LIKE for v in vals)
        has_dislike = any(v == Polarity.DISLIKE for v in vals)

        # If only one side appears, it's the clause polarity.
        if has_like and not has_dislike:
            return Polarity.LIKE
        if has_dislike and not has_like:
            return Polarity.DISLIKE

        # Mixed / ambiguous -> fall through.

    # Clause sentiment fallback if provided (POS/NEG only)
    if isinstance(clause_sentiment, str):
        s = clause_sentiment.strip().upper()
        if s == "POS":
            return Polarity.LIKE
        if s == "NEG":
            return Polarity.DISLIKE

    # Structural bias: elliptical neg fragments are treated as DISLIKE if still undecided
    if elliptical_neg:
        return Polarity.DISLIKE

    return Polarity.UNKNOWN


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
    """
    Does:
        Run a backend polarity function and normalize/patch its outputs.

    Returns:
        Mapping mention -> "LIKE"|"DISLIKE"|None
    """
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

        # Optional clause-level sentiment fallback if backend returns None
        if label is None and sent is not None:
            label = "LIKE" if sent == "POS" else "DISLIKE"

        # Structural bias: elliptical fragments default to DISLIKE if still None
        if elliptical_neg and label is None:
            label = "DISLIKE"

        out[m] = label

    return out


def infer_polarity_for_mentions_enum(
    clause_text: str,
    mentions: List[str],
    *,
    llm_polarity_fn: PolarityLLM,
    clause_sentiment: Optional[str] = None,
    elliptical_neg: bool = False,
) -> Dict[str, Polarity]:
    """
    Does:
        Same as infer_polarity_for_mentions(), but returns enum-typed polarities.

    Returns:
        Mapping mention -> Polarity (LIKE/DISLIKE/UNKNOWN)
    """
    raw = infer_polarity_for_mentions(
        clause_text,
        mentions,
        llm_polarity_fn=llm_polarity_fn,
        clause_sentiment=clause_sentiment,
        elliptical_neg=elliptical_neg,
    )
    return {m: _label_to_enum(v) for m, v in raw.items()}


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

    torch = require(
        "torch",
        extra="torch",
        purpose="Needed for make_free_polarity_fn() to select device and avoid meta-tensor loading.",
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

    # ✅ FIX: safe loading (prevents meta-tensor crash)
    encoder = _load_sentence_transformer_safe(
        SentenceTransformer=SentenceTransformer,
        model_name=model_name,
        torch=torch,
    )

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
    "infer_polarity_for_mentions_enum",
    "decide_clause_polarity",
    "make_free_polarity_fn",
]
