# src/Shopping_assistant/nlp/axis_classifier_embed.py
from __future__ import annotations

"""
Axis classification for constraint labels (e.g., "bright", "muddy") into:
BRIGHTNESS / SATURATION / VIBRANCY / DEPTH / CLARITY

Design (offline / deterministic):
- No paid APIs.
- Semantic inference via SentenceTransformer embeddings.
- Conservative thresholds: if unsure -> axis=None.

Contract:
    axis_fn(label, context="") -> AxisPred(label, axis|None, confidence, margin)
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from Shopping_assistant.nlp.schema import Axis
from Shopping_assistant.utils.optional_deps import require

log = logging.getLogger(__name__)


class AxisClassifier(Protocol):
    def __call__(self, label: str, *, context: str = "") -> "AxisPred":
        ...


@dataclass(frozen=True, slots=True)
class AxisPred:
    label: str
    axis: Optional[Axis]
    confidence: float
    margin: float
    meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Prototypes (small & conceptual; do NOT grow into a thesaurus)
# ---------------------------------------------------------------------------

_AXIS_PROTOTYPES: Dict[Axis, Tuple[str, ...]] = {
    Axis.BRIGHTNESS: (
        "bright",
        "light",
        "pale",
        "washed out",
        "too bright",
        "not too bright",
    ),
    Axis.DEPTH: (
        "dark",
        "deep",
        "rich",
        "very dark",
        "not too dark",
    ),
    Axis.SATURATION: (
        "saturated",
        "intense",
        "vivid",
        "high saturation",
        "too intense",
    ),
    Axis.VIBRANCY: (
        "neon",
        "vibrant",
        "electric",
        "fluorescent",
        "too neon",
    ),
    Axis.CLARITY: (
        "crisp",
        "muddy",
        "soft",
        "hazy",
        "clear",
        "not muddy",
    ),
}

_PROTOTYPES_VERSION = "v1"


def _canon_label(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _l2_normalize_rows(np: Any, X: Any, *, eps: float = 1e-12) -> Any:
    if getattr(X, "ndim", None) != 2:
        raise RuntimeError(f"Expected 2D array, got ndim={getattr(X, 'ndim', None)}")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + float(eps))


def _cosine_sim_row(np: Any, a: Any, B: Any) -> Any:
    # a: (d,), B: (k,d) -> (k,)
    return B @ a


@lru_cache(maxsize=8)
def make_axis_classifier_fn(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    min_sim: float = 0.35,
    min_margin: float = 0.08,
    debug: bool = False,
) -> AxisClassifier:
    """
    Does:
        Build an offline axis classifier using embedding similarity to axis prototypes.
    """
    np = require("numpy", extra="numpy", purpose="axis embedding classifier")
    st = require("sentence_transformers", extra="sentence-transformers", purpose="axis embedding classifier")
    SentenceTransformer = getattr(st, "SentenceTransformer", None)
    if SentenceTransformer is None:
        raise RuntimeError("sentence_transformers.SentenceTransformer not found")

    model = SentenceTransformer(model_name)

    axes: List[Axis] = list(_AXIS_PROTOTYPES.keys())
    axis_texts: List[str] = []
    axis_offsets: List[Tuple[int, int]] = []

    cur = 0
    for ax in axes:
        protos = list(_AXIS_PROTOTYPES[ax])
        axis_texts.extend(protos)
        axis_offsets.append((cur, cur + len(protos)))
        cur += len(protos)

    P = model.encode(axis_texts, convert_to_numpy=True, show_progress_bar=False)
    P = _l2_normalize_rows(np, P)

    @lru_cache(maxsize=4096)
    def _encode_one(text: str) -> Any:
        v = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        v = v / (np.linalg.norm(v) + 1e-12)
        return v

    def _predict(label: str, *, context: str = "") -> AxisPred:
        lab = _canon_label(label)
        if not lab:
            return AxisPred(label=lab, axis=None, confidence=0.0, margin=0.0, meta={"reason": "empty"})

        # Primary signal: label only (cache-friendly)
        v = _encode_one(lab)

        # Optional tie-break: add context only if close call (keeps cache stable)
        ctx = _canon_label(context)
        if ctx:
            v_ctx = _encode_one(f"{lab} | {ctx}")

        sims_axis: Dict[Axis, float] = {}
        for ax, (a0, a1) in zip(axes, axis_offsets):
            sims = _cosine_sim_row(np, v, P[a0:a1])
            sims_axis[ax] = float(np.max(sims))

        # sort
        ranked = sorted(sims_axis.items(), key=lambda x: x[1], reverse=True)
        best_ax, best_sim = ranked[0]
        second_sim = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = float(best_sim - second_sim)

        # If borderline, use context vector to re-rank (optional)
        if ctx and (best_sim < (min_sim + 0.05) or margin < (min_margin + 0.03)):
            sims_axis_ctx: Dict[Axis, float] = {}
            for ax, (a0, a1) in zip(axes, axis_offsets):
                sims = _cosine_sim_row(np, v_ctx, P[a0:a1])
                sims_axis_ctx[ax] = float(np.max(sims))
            ranked2 = sorted(sims_axis_ctx.items(), key=lambda x: x[1], reverse=True)
            best_ax2, best_sim2 = ranked2[0]
            second_sim2 = ranked2[1][1] if len(ranked2) > 1 else -1.0
            margin2 = float(best_sim2 - second_sim2)

            # adopt if it improves confidence or separation
            if best_sim2 > best_sim + 0.01 or margin2 > margin + 0.01:
                best_ax, best_sim, margin = best_ax2, float(best_sim2), float(margin2)
                sims_axis = sims_axis_ctx
                ranked = ranked2

        if best_sim < float(min_sim) or margin < float(min_margin):
            return AxisPred(
                label=lab,
                axis=None,
                confidence=float(best_sim),
                margin=float(margin),
                meta={
                    "prototypes_version": _PROTOTYPES_VERSION,
                    "best_axis": best_ax.value,
                    "ranked": [(a.value, float(s)) for a, s in ranked[:3]],
                    "reason": "below_threshold",
                }
                if debug
                else {"reason": "below_threshold"},
            )

        return AxisPred(
            label=lab,
            axis=best_ax,
            confidence=float(best_sim),
            margin=float(margin),
            meta={
                "prototypes_version": _PROTOTYPES_VERSION,
                "ranked": [(a.value, float(s)) for a, s in ranked[:3]],
            }
            if debug
            else {},
        )

    return _predict
