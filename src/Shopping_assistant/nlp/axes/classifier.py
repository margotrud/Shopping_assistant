# src/Shopping_assistant/nlp/axes/classifier.py
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
from typing import Any, Dict, List, Optional, Protocol, Tuple

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
# Axis "anchors": descriptions, not synonym lists (keeps things dynamic)
# ---------------------------------------------------------------------------

# Incorrect (removed): prototype lists containing degree/negation words (too/not/very).
# We classify *axis only* here. Direction/strength/negation are handled upstream in constraints.py.
_AXIS_DESCRIPTIONS: Dict[Axis, Tuple[str, ...]] = {
    Axis.BRIGHTNESS: (
        "Brightness / lightness of a lipstick shade (how light vs pale vs bright it looks).",
        "Perceived lightness: pale or washed-out vs bright and light.",
    ),
    Axis.DEPTH: (
        "Depth of a lipstick shade (how dark and rich it feels).",
        "Perceived depth: deep/rich/dark vs not deep.",
    ),
    Axis.SATURATION: (
        "Saturation/chroma of a lipstick shade (muted/desaturated vs saturated/intense).",
        "How colorful vs muted the shade is (desaturated vs saturated).",
    ),
    Axis.VIBRANCY: (
        "Vibrancy/neonness of a lipstick shade (neon/electric/fluorescent/flashy vs not neon).",
        "How punchy/flashy the color feels (neon/fluorescent vs quiet).",
    ),
    Axis.CLARITY: (
        "Clarity/cleanliness/softness of a lipstick shade (crisp/clear/clean vs soft/hazy/muddy).",
        "How clean vs muddy/soft/hazy the shade appears.",
    ),
}

_PROTOTYPES_VERSION = "v3_axis_descriptions"


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


def _load_sentence_transformer_safe(*, SentenceTransformer: Any, model_name: str, torch: Any) -> Any:
    device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"

    # 1) Robust path: build ST modules explicitly
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
        # 2) Fallback: standard constructor
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
        Build an offline axis classifier using embedding similarity to axis descriptions.

    Important:
        This classifier predicts the *axis only*. Negation/degree/direction are handled upstream.
    """
    np = require("numpy", extra="numpy", purpose="axis embedding classifier")
    torch = require("torch", extra="torch", purpose="axis embedding classifier (device selection)")
    st = require("sentence_transformers", extra="sentence-transformers", purpose="axis embedding classifier")
    SentenceTransformer = getattr(st, "SentenceTransformer", None)
    if SentenceTransformer is None:
        raise RuntimeError("sentence_transformers.SentenceTransformer not found")

    model = _load_sentence_transformer_safe(
        SentenceTransformer=SentenceTransformer,
        model_name=model_name,
        torch=torch,
    )

    axes: List[Axis] = list(_AXIS_DESCRIPTIONS.keys())
    axis_texts: List[str] = []
    axis_offsets: List[Tuple[int, int]] = []

    cur = 0
    for ax in axes:
        texts = list(_AXIS_DESCRIPTIONS[ax])
        axis_texts.extend(texts)
        axis_offsets.append((cur, cur + len(texts)))
        cur += len(texts)

    # Normalize embeddings once (faster + stable cosine)
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
        v_ctx = None
        if ctx:
            v_ctx = _encode_one(f"{lab} | {ctx}")

        sims_axis: Dict[Axis, float] = {}
        for ax, (a0, a1) in zip(axes, axis_offsets):
            sims = _cosine_sim_row(np, v, P[a0:a1])
            sims_axis[ax] = float(np.max(sims))

        ranked = sorted(sims_axis.items(), key=lambda x: x[1], reverse=True)
        best_ax, best_sim = ranked[0]
        second_sim = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = float(best_sim - second_sim)

        # If borderline, use context vector to re-rank (optional)
        if v_ctx is not None and (best_sim < (min_sim + 0.05) or margin < (min_margin + 0.03)):
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
