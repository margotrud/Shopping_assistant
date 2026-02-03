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
# Axis "anchors": conceptual descriptions (NO keyword lists)
# ---------------------------------------------------------------------------

_AXIS_DESCRIPTIONS: Dict[Axis, Tuple[str, ...]] = {
    Axis.BRIGHTNESS: (
        "Perceived lightness of a color: where it lies on the light ↔ dark continuum.",
        "How light or dark a shade appears (lightness), independent of chroma or neon feel.",
        "A shade that looks pale/light vs a shade that looks dim/dark, regardless of saturation.",
    ),
    Axis.DEPTH: (
        "Color depth as darkness/inkiness: how deep, dark, inky or dense a shade feels.",
        "Perceived darkness and density of a shade (inky/dense), distinct from saturation and neon pop.",
        "A shade that feels deep/inky/blackened vs a shade that feels light/pale, even if both are saturated.",
    ),
    Axis.SATURATION: (
        "Saturation/chroma: muted/desaturated ↔ saturated (strength of chroma), not neon/fluorescent.",
        "How saturated vs muted a shade appears, separate from lightness and from neon punch.",
        "A color that is washed-out/greyed vs a color that is fully chromatic, without implying neon.",
    ),
    Axis.VIBRANCY: (
        "Vibrancy/neon-pop: electric/fluorescent/flashy appearance (pop), not lightness and not darkness.",
        "Neon/electric/fluorescent look: a color that pops off the surface, distinct from mere saturation.",
        "A color with an electric/fluorescent 'pop' and punchiness, independent of being light or dark.",
        "Perceived pop/flashiness that can exist at many lightness levels (not a lightness axis).",
    ),
    Axis.CLARITY: (
        "Clarity/cleanliness vs hazy/muddy/soft appearance of a shade.",
        "How clean/crisp vs muddy/hazy a shade looks, independent of brightness and saturation.",
        "A shade that looks clear/sharp vs a shade that looks cloudy/soft/murky.",
    ),
}

_PROTOTYPES_VERSION = "v8_add_brightness_vibrancy_tiebreak"


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


def _maybe_get_wordnet():
    """
    Returns nltk.corpus.wordnet if available, else None.
    No hard dependency.
    """
    try:
        nltk = require("nltk", extra="nltk", purpose="optional WordNet sense rerank for axis classification")
        wn = getattr(getattr(nltk, "corpus", None), "wordnet", None)
        return wn
    except Exception:
        return None


def _wordnet_adj_synset_texts(wn: Any, label: str) -> List[str]:
    """
    Build texts from adjective synsets: definition + examples.
    This is not a synonym table; it's sense descriptions.
    """
    out: List[str] = []
    try:
        synsets = wn.synsets(label, pos=getattr(wn, "ADJ", None) or "a")
    except Exception:
        synsets = []

    for ss in synsets or []:
        try:
            d = ss.definition() or ""
        except Exception:
            d = ""
        try:
            ex = ss.examples() or []
        except Exception:
            ex = []
        txt = " ".join([str(d).strip()] + [str(e).strip() for e in ex if str(e).strip()])
        txt = _canon_label(txt)
        if txt:
            out.append(txt)

    # fallback: allow satellites ('s') if available
    if not out:
        try:
            synsets = wn.synsets(label, pos=getattr(wn, "ADJ_SAT", None) or "s")
        except Exception:
            synsets = []
        for ss in synsets or []:
            try:
                d = ss.definition() or ""
            except Exception:
                d = ""
            try:
                ex = ss.examples() or []
            except Exception:
                ex = []
            txt = " ".join([str(d).strip()] + [str(e).strip() for e in ex if str(e).strip()])
            txt = _canon_label(txt)
            if txt:
                out.append(txt)

    return out


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

    def _score_axes_from_vec(v_any: Any) -> Dict[Axis, float]:
        sims_axis: Dict[Axis, float] = {}
        for ax, (a0, a1) in zip(axes, axis_offsets):
            sims = _cosine_sim_row(np, v_any, P[a0:a1])
            sims_axis[ax] = float(np.max(sims))
        return sims_axis

    def _cos(a: Any, b: Any) -> float:
        return float(a @ b)

    # -----------------------------------------------------------------------
    # Contrastive tie-break probes (semantic; not token maps)
    # -----------------------------------------------------------------------
    @lru_cache(maxsize=256)
    def _probe_vec(text: str) -> Any:
        v = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        v = v / (np.linalg.norm(v) + 1e-12)
        return v

    def _tie_break_lightness(v_label: Any) -> Optional[Axis]:
        # brightness vs depth (lightness pole test)
        v_bright = _probe_vec("high lightness; light; pale; opposite of dark")
        v_dark = _probe_vec("low lightness; dark; deep; inky; opposite of light")
        s_b = _cos(v_label, v_bright)
        s_d = _cos(v_label, v_dark)
        if abs(s_b - s_d) < 0.02:
            return None
        return Axis.BRIGHTNESS if s_b > s_d else Axis.DEPTH

    def _tie_break_brightness_vibrancy(v_label: Any) -> Optional[Axis]:
        # lightness vs neon-pop (prevents 'bright' drifting to vibrancy)
        v_bri = _probe_vec("lightness; amount of light; pale/light; opposite of dark")
        v_vib = _probe_vec("neon; fluorescent; electric; flashy; color pops; not about lightness")
        s_b = _cos(v_label, v_bri)
        s_v = _cos(v_label, v_vib)
        if abs(s_b - s_v) < 0.02:
            return None
        return Axis.BRIGHTNESS if s_b > s_v else Axis.VIBRANCY

    def _tie_break_vibrancy_depth(v_label: Any) -> Optional[Axis]:
        # vibrancy vs depth (pop vs darkness)
        v_vib = _probe_vec("neon; fluorescent; electric; flashy; vivid pop; eye-catching")
        v_dep = _probe_vec("deep; dark; inky; dense; blackened; low lightness")
        s_v = _cos(v_label, v_vib)
        s_d = _cos(v_label, v_dep)
        if abs(s_v - s_d) < 0.02:
            return None
        return Axis.VIBRANCY if s_v > s_d else Axis.DEPTH

    # -----------------------------------------------------------------------
    # Optional WordNet sense rerank for brightness vs depth ambiguity
    # -----------------------------------------------------------------------
    def _wordnet_sense_rerank_if_needed(
        label_clean: str,
        ranked: List[Tuple[Axis, float]],
        *,
        context: str,
    ) -> Optional[Tuple[Axis, float, float, Dict[str, Any]]]:
        """
        Non-hardcoded disambiguation for brightness vs depth using WordNet sense descriptions.
        Runs only when top-2 includes {BRIGHTNESS, DEPTH} and separation is weak.
        """
        if len(ranked) < 2:
            return None

        top2 = {ranked[0][0], ranked[1][0]}
        if top2 != {Axis.BRIGHTNESS, Axis.DEPTH}:
            return None

        best_sim = float(ranked[0][1])
        second_sim = float(ranked[1][1])
        margin = float(best_sim - second_sim)

        # only when ambiguous
        if best_sim >= (float(min_sim) + 0.10) and margin >= (float(min_margin) + 0.05):
            return None

        wn = _maybe_get_wordnet()
        if wn is None:
            return None

        syn_texts = _wordnet_adj_synset_texts(wn, label_clean)
        if not syn_texts:
            return None

        ctx = _canon_label(context)
        use_texts = [f"{label_clean} | {t} | {ctx}" if ctx else f"{label_clean} | {t}" for t in syn_texts]

        per_ax_best: Dict[str, float] = {}
        for txt in use_texts:
            v = _encode_one(txt)
            sims = _score_axes_from_vec(v)
            b = float(sims.get(Axis.BRIGHTNESS, -1.0))
            d = float(sims.get(Axis.DEPTH, -1.0))
            per_ax_best["brightness"] = max(per_ax_best.get("brightness", -1.0), b)
            per_ax_best["depth"] = max(per_ax_best.get("depth", -1.0), d)

        b_best = float(per_ax_best.get("brightness", -1.0))
        d_best = float(per_ax_best.get("depth", -1.0))
        if b_best == d_best:
            return None

        best_ax = Axis.BRIGHTNESS if b_best > d_best else Axis.DEPTH
        margin_r = float(abs(b_best - d_best))

        meta = {
            "source": "wordnet_synset_definition_rerank",
            "wordnet_n_synsets": int(len(syn_texts)),
            "wordnet_best_scores": {"brightness": b_best, "depth": d_best},
        }
        return best_ax, float(max(b_best, d_best)), margin_r, meta

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

        sims_axis = _score_axes_from_vec(v)
        ranked = sorted(sims_axis.items(), key=lambda x: x[1], reverse=True)
        best_ax, best_sim = ranked[0]
        second_sim = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = float(best_sim - second_sim)

        # If borderline, use context vector to re-rank (optional)
        if v_ctx is not None and (best_sim < (min_sim + 0.05) or margin < (min_margin + 0.03)):
            sims_axis_ctx = _score_axes_from_vec(v_ctx)
            ranked2 = sorted(sims_axis_ctx.items(), key=lambda x: x[1], reverse=True)
            best_ax2, best_sim2 = ranked2[0]
            second_sim2 = ranked2[1][1] if len(ranked2) > 1 else -1.0
            margin2 = float(best_sim2 - second_sim2)

            # adopt if it improves confidence or separation
            if best_sim2 > best_sim + 0.01 or margin2 > margin + 0.01:
                best_ax, best_sim, margin = best_ax2, float(best_sim2), float(margin2)
                ranked = ranked2

        # Tie-breakers for close calls (semantic probes; not keyword rules)
        if best_sim >= float(min_sim) and margin < float(min_margin) and len(ranked) >= 2:
            ax1, ax2 = ranked[0][0], ranked[1][0]
            top2 = {ax1, ax2}

            # brightness vs depth
            if top2 == {Axis.BRIGHTNESS, Axis.DEPTH}:
                chosen = _tie_break_lightness(v)
                if chosen is not None:
                    best_ax = chosen
                    margin = float(min_margin)

            # brightness vs vibrancy (prevents 'bright' -> vibrancy)
            elif top2 == {Axis.BRIGHTNESS, Axis.VIBRANCY}:
                chosen = _tie_break_brightness_vibrancy(v)
                if chosen is not None:
                    best_ax = chosen
                    margin = float(min_margin)

            # vibrancy vs depth (prevents 'vibrant' drifting to depth)
            elif top2 == {Axis.VIBRANCY, Axis.DEPTH}:
                chosen = _tie_break_vibrancy_depth(v)
                if chosen is not None:
                    best_ax = chosen
                    margin = float(min_margin)

        # Optional WordNet sense rerank ONLY for brightness vs depth ambiguity
        wn_r = _wordnet_sense_rerank_if_needed(lab, ranked, context=context)
        adopted_wordnet = False
        meta_wordnet: Dict[str, Any] = {}
        if wn_r is not None:
            ax_r, sim_r, margin_r, meta_r = wn_r
            if float(sim_r) >= float(best_sim) - 0.03:
                best_ax = ax_r
                best_sim = float(max(best_sim, sim_r))
                margin = float(max(margin, margin_r))
                adopted_wordnet = True
                meta_wordnet = meta_r

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
                    **({"wordnet": meta_wordnet} if adopted_wordnet else {}),
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
                **({"wordnet": meta_wordnet} if adopted_wordnet else {}),
            }
            if debug
            else {},
        )

    return _predict


__all__ = ["AxisPred", "make_axis_classifier_fn"]
