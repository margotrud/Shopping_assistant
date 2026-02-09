# src/Shopping_assistant/nlp/polarity.py
from __future__ import annotations

"""
Polarity classification for mentions (LIKE / DISLIKE / UNKNOWN).

Backends:
- lexical: fast, deterministic (Streamlit default via SA_POLARITY_BACKEND=lexical)
- semantic: SentenceTransformer embeddings (offline default)

Import-safety:
- numpy / torch / sentence_transformers are imported lazily and only when using semantic backend.
"""

import logging
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Protocol

from Shopping_assistant.nlp.schema import Polarity
from Shopping_assistant.utils.optional_deps import require

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------


class PolarityLLM(Protocol):
    """Does: polarity helper for clauses/mentions (backend-agnostic)."""

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


def _env_csv(key: str) -> List[str]:
    v = os.environ.get(key, "").strip()
    if not v:
        return []
    return [x.strip().lower() for x in v.split(",") if x.strip()]


def _env_choice(key: str, default: str) -> str:
    v = os.environ.get(key, "").strip().lower()
    return v or default.strip().lower()


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
        Decide a clause-level polarity from mention labels + optional sentiment + structural bias.
    """
    if mention_labels:
        vals = [_label_to_enum(v) for v in mention_labels.values() if v is not None]
        has_like = any(v == Polarity.LIKE for v in vals)
        has_dislike = any(v == Polarity.DISLIKE for v in vals)

        if has_like and not has_dislike:
            return Polarity.LIKE
        if has_dislike and not has_like:
            return Polarity.DISLIKE

    if isinstance(clause_sentiment, str):
        s = clause_sentiment.strip().upper()
        if s == "POS":
            return Polarity.LIKE
        if s == "NEG":
            return Polarity.DISLIKE

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
        Run backend polarity fn and normalize outputs; adds sentiment + structural fallbacks.
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

        if label is None and sent is not None:
            label = "LIKE" if sent == "POS" else "DISLIKE"

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
        Same as infer_polarity_for_mentions(), returning enum-typed polarities.
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
# Fast lexical backend (no torch / no embeddings)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def make_lexical_polarity_fn(*, debug: bool = False) -> PolarityLLM:
    """
    Does:
        Build a cheap polarity function using negation + intent triggers (env-overridable).
    """
    # Minimal defaults; extend via env to avoid code edits.
    pos_trig = _env_csv("SA_POLARITY_POS_TRIGGERS") or [
        "want",
        "like",
        "love",
        "prefer",
        "looking for",
        "need",
        "give me",
    ]
    neg_trig = _env_csv("SA_POLARITY_NEG_TRIGGERS") or [
        "not",
        "don't",
        "dont",
        "avoid",
        "without",
        "no",
        "exclude",
    ]

    # Phrase-first matching (longer first) with soft boundaries.
    def _compile(xs: List[str]) -> List[re.Pattern]:
        pats: List[re.Pattern] = []
        for x in sorted({s.strip().lower() for s in xs if s and s.strip()}, key=len, reverse=True):
            esc = re.escape(x)
            pats.append(re.compile(rf"(?i)(?:^|[\s,.;:!?()]){esc}(?:$|[\s,.;:!?()])"))
        return pats

    POS = _compile(pos_trig)
    NEG = _compile(neg_trig)

    def _has_any(pats: List[re.Pattern], t: str) -> bool:
        return any(p.search(t) for p in pats)

    def _fn(clause_text: str, mentions: List[str]) -> Dict[str, Optional[str]]:
        if not mentions:
            return {}

        t = f" {clause_text.strip()} "
        neg = _has_any(NEG, t)
        pos = _has_any(POS, t)

        if neg and not pos:
            lab: Optional[str] = "DISLIKE"
        elif pos and not neg:
            lab = "LIKE"
        else:
            lab = None

        if debug:
            log.debug("[polarity][lex] pos=%s neg=%s -> %s text=%r", pos, neg, lab, clause_text)

        return {m: lab for m in mentions}

    return _fn


# ---------------------------------------------------------------------------
# Offline semantic backend (SentenceTransformer)
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
    Does:
        Semantic polarity via embeddings; returns None when ambiguous (min_sim/min_margin).
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

    min_sim_f = max(0.0, float(min_sim))
    min_margin_f = max(0.0, float(min_margin))

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

        return {m: out_u.get(m) for m in mentions}

    return _fn


# ---------------------------------------------------------------------------
# Backend dispatcher
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def make_polarity_fn(
    *,
    backend: Optional[str] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_sim: float = 0.30,
    min_margin: float = 0.10,
    debug: bool = False,
) -> PolarityLLM:
    """
    Does:
        Return a polarity backend. backend is read from SA_POLARITY_BACKEND when None.
    """
    b = (backend or _env_choice("SA_POLARITY_BACKEND", "semantic")).strip().lower()
    if b in {"lex", "lexical", "rules"}:
        return make_lexical_polarity_fn(debug=debug)
    if b in {"free", "semantic", "st", "sentence-transformer"}:
        return make_free_polarity_fn(model_name=model_name, min_sim=min_sim, min_margin=min_margin, debug=debug)
    raise ValueError(f"Unknown polarity backend: {b!r} (expected: lexical|semantic)")


__all__ = [
    "PolarityLLM",
    "infer_polarity_for_mentions",
    "infer_polarity_for_mentions_enum",
    "decide_clause_polarity",
    "make_lexical_polarity_fn",
    "make_free_polarity_fn",
    "make_polarity_fn",
]
