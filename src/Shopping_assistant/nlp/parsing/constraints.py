# src/Shopping_assistant/nlp/parsing/constraints.py
"""
Constraint parsing logic.

Extracts structured constraint signals (axis, direction, strength)
from NLP outputs for downstream filtering and scoring.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

from Shopping_assistant.nlp.axes.predictor import predict_axis
from Shopping_assistant.nlp.schema import Axis, Constraint, Direction, Polarity, Strength
from Shopping_assistant.utils.optional_deps import require

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Token
else:
    Language = Any  # type: ignore
    Doc = Any  # type: ignore
    Token = Any  # type: ignore


# ---------------------------------------------------------------------
# Axis mapping
# ---------------------------------------------------------------------

_AXIS_LEXICON: Dict[str, Axis] = {
    # Hard override for known ambiguous token(s)
    "dim": Axis.BRIGHTNESS,
    # Cosmetics/domain frequent: ensure stable mapping (NOT per-colors hardcoding)
    "muted": Axis.SATURATION,
    "dusty": Axis.SATURATION,
    "saturated": Axis.SATURATION,
    "desaturated": Axis.SATURATION,

}

_COLOR_DOMAIN_NOUNS = {
    "colors",
    "shade",
    "tone",
    "tint",
    "hue",
    "lipstick",
    "gloss",
    "balm",
    "liner",
    "stain",
}

_AXIS_FAMILY: Dict[Axis, str] = {
    Axis.BRIGHTNESS: "LIGHTNESS",
    Axis.DEPTH: "LIGHTNESS",
    Axis.SATURATION: "CHROMA",
    Axis.VIBRANCY: "CHROMA",
    Axis.CLARITY: "SURFACE",
}

_FAMILY_AXES: Dict[str, set[Axis]] = {}
for ax, fam in _AXIS_FAMILY.items():
    _FAMILY_AXES.setdefault(fam, set()).add(ax)


def _normalize_model_name(model_name: str) -> str:
    if "/" in (model_name or ""):
        return model_name
    return f"sentence-transformers/{model_name}"


def _norm_text_piece(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"n't", "nt"}:
        return "not"
    return s


def _clean_axis_core(tok: Token) -> str:
    """
    Keep surface form for adjectival participles (spaCy often tags them as VERB/VBN):
    "muted", "toned", "sheered" should NOT become "mute/tone/sheer" for axis mapping.
    """
    raw = _norm_text_piece(tok.text)
    lem = _norm_text_piece(tok.lemma_)

    if tok.dep_ == "amod" and tok.tag_ in {"VBN", "VBD"}:
        return raw or lem

    return lem or raw


def _iter_axis_queries(tok: Token) -> List[str]:
    lemma = _clean_axis_core(tok)
    out: List[str] = []
    if lemma:
        out.append(lemma)
        out.append(f"{lemma} colors")

    seen = set()
    uniq: List[str] = []
    for q in out:
        q = " ".join((q or "").split())
        if q and q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


def _axis_from_text(
    text: str,
    *,
    mapper_model: str,
    mapper_threshold: float,
    mapper_min_margin: float,
    axis_source: str,
    debug: bool = False,
) -> Tuple[Optional[Axis], Dict[str, Any], Any]:
    pred = predict_axis(
        text,
        context="cosmetics colors attribute",
        model_name=_normalize_model_name(mapper_model),
        min_sim=float(mapper_threshold),
        min_margin=float(mapper_min_margin),
        debug=bool(debug),
    )

    meta: Dict[str, Any] = {
        "axis_source": axis_source,
        "axis_score": float(getattr(pred, "confidence", 0.0)),
        "axis_margin": float(getattr(pred, "margin", 0.0)),
        "axis": pred.axis.value if getattr(pred, "axis", None) is not None else None,
        "axis_query": text,
    }

    if getattr(pred, "axis", None) is not None:
        meta["axis_gate"] = "STRICT"

    if debug:
        meta["axis_debug"] = getattr(pred, "meta", None)
    return pred.axis, meta, pred


def _ranked_axes_from_pred(pred: Any) -> List[Tuple[Axis, float]]:
    meta = getattr(pred, "meta", None)
    if not isinstance(meta, dict):
        return []

    ranked = meta.get("ranked") or meta.get("ranked_top5") or meta.get("ranked_top3")
    if not ranked:
        return []

    out: List[Tuple[Axis, float]] = []
    for a, s in ranked:
        try:
            ax = Axis(a) if isinstance(a, str) else a
            out.append((ax, float(s)))
        except Exception:
            continue
    return out


def _best_axis_from_pred(pred: Any) -> Optional[Axis]:
    ax = getattr(pred, "axis", None)
    if ax is not None:
        return ax

    meta = getattr(pred, "meta", None)
    if isinstance(meta, dict):
        ba = meta.get("best_axis")
        if isinstance(ba, str):
            try:
                return Axis(ba)
            except Exception:
                return None
    return None


def _context_is_colorish(tok: Token, *, window: int = 5) -> bool:
    head = getattr(tok, "head", None)
    if head is not None and head.pos_ in {"NOUN", "PROPN"} and _norm_text_piece(head.lemma_) in _COLOR_DOMAIN_NOUNS:
        return True

    doc = tok.doc
    for j in range(max(0, tok.i - window), min(len(doc), tok.i + window + 1)):
        t = doc[j]
        if t.pos_ in {"NOUN", "PROPN"} and _norm_text_piece(t.lemma_) in _COLOR_DOMAIN_NOUNS:
            return True
    return False


def _family(ax: Axis) -> str:
    return _AXIS_FAMILY.get(ax, "OTHER")


def _best_lightness_from_ranked(ranked: List[Tuple[Axis, float]]) -> Optional[Tuple[Axis, float]]:
    cand = [(a, s) for a, s in ranked if _family(a) == "LIGHTNESS"]
    if not cand:
        return None
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[0]


def _family_fallback(
    pred: Any,
    *,
    min_sim: float,
    min_margin: float,
    color_context: bool,
) -> Tuple[Optional[Axis], Dict[str, Any]]:
    conf = float(getattr(pred, "confidence", 0.0))
    margin = float(getattr(pred, "margin", 0.0))
    best_ax = _best_axis_from_pred(pred)

    meta: Dict[str, Any] = {}
    if best_ax is None:
        return None, meta

    if conf < float(min_sim):
        meta["axis_gate"] = "FAIL_MIN_SIM"
        return None, meta

    if margin >= float(min_margin):
        meta["axis_gate"] = "STRICT"
        return best_ax, meta

    ranked = _ranked_axes_from_pred(pred)
    if len(ranked) < 2:
        meta["axis_gate"] = "FAIL_NO_RANKED"
        return None, meta

    ax1, s1 = ranked[0]
    ax2, s2 = ranked[1]
    fam1 = _family(ax1)
    fam2 = _family(ax2)

    family_margin_floor = 0.02 if color_context else 0.0
    if margin < family_margin_floor:
        meta.update(
            {
                "axis_gate": "FAIL_MARGIN_TOO_FLAT",
                "axis_family_top1": fam1,
                "axis_family_top2": fam2,
                "axis_top2": [(ax1.value, float(s1)), (ax2.value, float(s2))],
                "axis_color_context": bool(color_context),
            }
        )
        return None, meta

    if fam1 == fam2 and fam1 in {"LIGHTNESS", "CHROMA"}:
        chosen_ax = ax1
        chosen_reason = "FAMILY"

        if fam1 == "CHROMA":
            light = _best_lightness_from_ranked(ranked[:3])
            if light is not None:
                axL, sL = light
                if float(sL) >= float(s1) - 0.03:
                    chosen_ax = axL
                    chosen_reason = "FAMILY_BIAS_LIGHTNESS"

        meta.update(
            {
                "axis_gate": chosen_reason,
                "axis_family": _family(chosen_ax),
                "axis_family_top1": fam1,
                "axis_family_top2": fam2,
                "axis_ambiguous": True,
                "axis_top2": [(ax1.value, float(s1)), (ax2.value, float(s2))],
                "axis_margin_used": float(margin),
                "axis_conf_used": float(conf),
                "axis_color_context": bool(color_context),
            }
        )
        return chosen_ax, meta

    meta.update(
        {
            "axis_gate": "FAIL_FAMILY",
            "axis_family_top1": fam1,
            "axis_family_top2": fam2,
            "axis_top2": [(ax1.value, float(s1)), (ax2.value, float(s2))],
            "axis_color_context": bool(color_context),
        }
    )
    return None, meta


def _axis_from_token(
    tok: Token,
    *,
    mapper_model: str = "all-MiniLM-L6-v2",
    mapper_threshold: float = 0.35,
    mapper_min_margin: float = 0.08,
) -> Tuple[Optional[Axis], Dict[str, Any]]:
    lemma = _norm_text_piece(tok.lemma_)
    base_meta: Dict[str, Any] = {"tok_lemma": lemma, "tok_pos": tok.pos_, "tok_text": tok.text}

    # --- LEXICON FIRST (hard override for known ambiguous tokens / frequent domain adjectives) ---
    axis2 = _AXIS_LEXICON.get(lemma)

    if axis2 is None and lemma.endswith("ness") and len(lemma) > 6:
        axis2 = _AXIS_LEXICON.get(lemma[:-4])  # dimness -> dim

    if axis2 is None and lemma.endswith("ed") and len(lemma) > 4:
        axis2 = _AXIS_LEXICON.get(lemma[:-2])  # dimmed -> dim

    if axis2 is not None:
        meta = {
            **base_meta,
            "axis_source": "lexicon_override",
            "axis_score": 1.0,
            "axis_margin": 1.0,
            "axis": axis2.value,
            "axis_query": lemma,
            "axis_gate": "LEXICON",
            "axis_query_strategy": "lexicon_first",
        }
        return axis2, meta

    # --- then embedding tries ---
    last_meta2: Dict[str, Any] = {}
    last_fb_meta: Dict[str, Any] = {}

    for q in _iter_axis_queries(tok):
        axis, meta2, pred = _axis_from_text(
            q,
            mapper_model=mapper_model,
            mapper_threshold=mapper_threshold,
            mapper_min_margin=mapper_min_margin,
            axis_source="embed",
            debug=True,
        )
        last_meta2 = dict(meta2)
        last_fb_meta = {}

        if axis is not None:
            meta = {
                **base_meta,
                **meta2,
                "axis_query_strategy": "multi_try_first_pass",
                "axis_gate": meta2.get("axis_gate", "STRICT"),
            }
            return axis, meta

        best_ax = _best_axis_from_pred(pred)
        if best_ax is not None:
            color_ctx = _context_is_colorish(tok)
            ax_fb, fb_meta = _family_fallback(
                pred,
                min_sim=mapper_threshold,
                min_margin=mapper_min_margin,
                color_context=color_ctx,
            )
            last_fb_meta = dict(fb_meta)
            if ax_fb is not None:
                meta = {
                    **base_meta,
                    **meta2,
                    **fb_meta,
                    "axis_query_strategy": "multi_try_family_fallback",
                    "axis": ax_fb.value,
                }
                return ax_fb, meta

    meta = {**base_meta, **last_meta2, **last_fb_meta, "axis_query_strategy": "multi_try_all_failed"}
    return None, meta


# ---------------------------------------------------------------------
# NON-STATIC negation + degree + axis-extremeness (NO noise-word lists)
# ---------------------------------------------------------------------


def _has_negation(tok: Token) -> bool:
    return any(ch.dep_ == "neg" for ch in tok.children)


def _advmods(tok: Token) -> List[Token]:
    out = [ch for ch in tok.children if ch.dep_ == "advmod" and ch.pos_ == "ADV"]
    return sorted({t.i: t for t in out}.values(), key=lambda t: t.i)


def _has_degree_morph(tok: Token) -> bool:
    try:
        deg = tok.morph.get("Degree") or []
        return any(d in {"Cmp", "Sup"} for d in deg)
    except Exception:
        return False


@lru_cache(maxsize=1)
def _get_st_model(model_name: str):
    st = require(
        "sentence_transformers",
        extra="sentence-transformers",
        purpose="constraints: non-static degree/negation scoring",
    )
    SentenceTransformer = getattr(st, "SentenceTransformer", None)
    if SentenceTransformer is None:
        raise ImportError("sentence_transformers.SentenceTransformer not found")
    return SentenceTransformer(_normalize_model_name(model_name))


def _cos_sim(a, b) -> float:
    denom = float((a**2).sum() ** 0.5) * float((b**2).sum() ** 0.5)
    if denom <= 0:
        return 0.0
    return float((a @ b) / denom)


@lru_cache(maxsize=256)
def _op_anchor_vecs(model_name: str):
    m = _get_st_model(model_name)
    v_neg = m.encode("negation (denial / absence)", normalize_embeddings=False)
    v_cap = m.encode("too much / excessive", normalize_embeddings=False)
    v_more = m.encode("increase / more", normalize_embeddings=False)
    v_less = m.encode("decrease / less", normalize_embeddings=False)
    return v_neg, v_cap, v_more, v_less


@lru_cache(maxsize=4096)
def _token_vec(token_lemma: str, *, model_name: str):
    m = _get_st_model(model_name)
    return m.encode((token_lemma or "").strip().lower(), normalize_embeddings=False)


def _is_neg_cue_token(t: Token, *, model_name: str) -> bool:
    if t.dep_ == "neg":
        return True
    v = _token_vec(_norm_text_piece(t.lemma_), model_name=model_name)
    v_neg, _, _, _ = _op_anchor_vecs(model_name)
    return _cos_sim(v, v_neg) >= 0.45


def _neg_cues_in_window(tok: Token, *, window: int, model_name: str) -> List[Token]:
    doc = tok.doc
    out: List[Token] = []
    for j in range(1, window + 1):
        i = tok.i - j
        if i < 0:
            break
        t = doc[i]
        if _is_neg_cue_token(t, model_name=model_name):
            out.append(t)
    return sorted({t.i: t for t in out}.values(), key=lambda t: t.i)


def _has_negation_anywhere(
    tok: Token,
    *,
    max_hops: int = 3,
    window: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
) -> bool:
    if _has_negation(tok):
        return True

    cur = tok
    for _ in range(max_hops):
        head = getattr(cur, "head", None)
        if head is None or head == cur:
            break
        if any(ch.dep_ == "neg" for ch in head.children):
            return True
        cur = head

    return bool(_neg_cues_in_window(tok, window=window, model_name=model_name))


def _degree_operator(tok: Token, *, model_name: str = "all-MiniLM-L6-v2", window: int = 4) -> Tuple[str, float]:
    """
    Returns (op, score) where op in {"CAP","MORE","LESS","NONE"}.
    """
    doc = tok.doc
    cands: List[Token] = []

    cands.extend(_advmods(tok))

    head = getattr(tok, "head", None)
    if head is not None and head is not tok:
        cands.extend(_advmods(head))

    for j in range(1, window + 1):
        i = tok.i - j
        if i < 0:
            break
        t = doc[i]
        lem = _norm_text_piece(t.lemma_) or (t.text or "").strip().lower()
        if t.pos_ == "ADV" or lem in {"too", "more", "less"}:
            cands.append(t)

    def _subtree_ops(root: Token) -> Optional[str]:
        for t in root.subtree:
            lem = _norm_text_piece(t.lemma_) or (t.text or "").strip().lower()
            if lem == "too":
                return "CAP"
            if lem == "more":
                return "MORE"
            if lem == "less":
                return "LESS"
        return None

    op_sub = _subtree_ops(tok)
    if op_sub is None and head is not None and head is not tok:
        op_sub = _subtree_ops(head)

    if op_sub is not None:
        return op_sub, 1.0

    cands = sorted({t.i: t for t in cands}.values(), key=lambda t: t.i)
    if not cands:
        return "NONE", 0.0

    for a in cands:
        lem = _norm_text_piece(a.lemma_) or (a.text or "").strip().lower()
        if lem == "too":
            return "CAP", 1.0
        if lem == "more":
            return "MORE", 1.0
        if lem == "less":
            return "LESS", 1.0

    _v_neg, v_cap, v_more, v_less = _op_anchor_vecs(model_name)
    best_op = "NONE"
    best_score = 0.0
    for a in cands:
        v = _token_vec(_norm_text_piece(a.lemma_), model_name=model_name)
        s_cap = _cos_sim(v, v_cap)
        s_more = _cos_sim(v, v_more)
        s_less = _cos_sim(v, v_less)
        op, sc = max((("CAP", s_cap), ("MORE", s_more), ("LESS", s_less)), key=lambda x: x[1])
        if sc > best_score:
            best_op, best_score = op, float(sc)

    if best_score < 0.45:
        return "NONE", float(best_score)
    return best_op, float(best_score)


@lru_cache(maxsize=256)
def _degree_pole_vecs(model_name: str):
    m = _get_st_model(model_name)
    low = m.encode("slightly / mildly", normalize_embeddings=False)
    high = m.encode("extremely / intensely", normalize_embeddings=False)
    return low, high


def _softmax2(a: float, b: float, *, t: float = 0.08) -> float:
    ea = math.exp(a / t)
    eb = math.exp(b / t)
    return eb / (ea + eb)


@lru_cache(maxsize=4096)
def _adv_intensity_score(adv_lemma: str, *, model_name: str) -> float:
    w = (adv_lemma or "").strip().lower()
    if not w:
        return 0.0

    v = _token_vec(w, model_name=model_name)
    low, high = _degree_pole_vecs(model_name)

    s_low = _cos_sim(v, low)
    s_high = _cos_sim(v, high)

    return float(_softmax2(s_low, s_high, t=0.08))


# ---------------------------------------------------------------------
# Axis poles (conceptual, not token lists)
# ---------------------------------------------------------------------


@lru_cache(maxsize=256)
def _axis_poles(model_name: str) -> Dict[Axis, Tuple[str, str, str]]:
    """
    Returns (low_pole_text, high_pole_text, neutral_text) for extremeness scoring.
    These are conceptual descriptions (not synonym lists).
    """
    return {
        Axis.BRIGHTNESS: (
            "lower perceived lightness; darker appearance; less light in the shade",
            "higher perceived lightness; lighter appearance; more light in the shade",
            "neither especially light nor dark; medium lightness",
        ),
        Axis.DEPTH: (
            "lower depth; less dense; less rich; less inky appearance",
            "higher depth; richer denser inkier appearance",
            "moderate depth; neither especially deep nor shallow",
        ),
        Axis.SATURATION: (
            "less saturated chroma; more muted or desaturated appearance",
            "more saturated chroma; less muted; stronger chroma",
            "moderate saturation; neither muted nor highly saturated",
        ),
        Axis.VIBRANCY: (
            "less neon/flashy; more subdued; not electric or fluorescent",
            "more neon/electric/fluorescent; flashier appearance",
            "moderate vibrancy; neither subdued nor neon",
        ),
        Axis.CLARITY: (
            "less clear; hazy or muddy; softer appearance",
            "clearer crisper cleaner appearance; less muddy",
            "moderate clarity; neither especially crisp nor muddy",
        ),
    }


@lru_cache(maxsize=4096)
def _text_vec(text: str, *, model_name: str):
    m = _get_st_model(model_name)
    return m.encode((text or "").strip().lower(), normalize_embeddings=False)


def _axis_extremeness(tok: Token, *, axis: Axis, model_name: str) -> Tuple[float, Direction]:
    lemma = _norm_text_piece(tok.lemma_)
    if not lemma:
        return 0.0, Direction.RAISE

    low, high, neu = _axis_poles(model_name)[axis]

    v = _text_vec(lemma, model_name=model_name)
    v_low = _text_vec(low, model_name=model_name)
    v_high = _text_vec(high, model_name=model_name)
    v_neu = _text_vec(neu, model_name=model_name)

    s_low = _cos_sim(v, v_low)
    s_high = _cos_sim(v, v_high)
    s_neu = _cos_sim(v, v_neu)

    best_pole = max(s_low, s_high)
    delta = best_pole - s_neu

    x = 8.0 * delta
    ext = float(1.0 / (1.0 + math.exp(-x)))

    direction = Direction.RAISE if s_high >= s_low else Direction.LOWER
    return max(0.0, min(1.0, ext)), direction


def _degree_score(
    tok: Token,
    *,
    axis: Optional[Axis] = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> float:
    # axis currently unused (kept for API consistency and future axis-aware tuning)
    if _has_negation_anywhere(tok, model_name=model_name):
        return 0.9
    if _has_degree_morph(tok):
        return 0.85

    base_word = _adv_intensity_score(_norm_text_piece(tok.lemma_), model_name=model_name)

    advs = _advmods(tok)
    if not advs:
        op, op_sc = _degree_operator(tok, model_name=model_name)
        if op != "NONE":
            return max(0.0, min(1.0, 0.45 + 0.45 * op_sc + 0.10 * base_word))
        return float(max(0.0, min(1.0, 0.35 * base_word)))

    base = 1.0 - math.exp(-0.9 * len(advs))
    sem = max(_adv_intensity_score(_norm_text_piece(a.lemma_), model_name=model_name) for a in advs)

    op, op_sc = _degree_operator(tok, model_name=model_name)
    op_boost = 0.0 if op == "NONE" else max(0.0, min(1.0, 0.25 + 0.35 * op_sc))

    return max(0.0, min(1.0, 0.25 * base + 0.50 * sem + 0.15 * op_boost + 0.10 * base_word))


def _has_degree_anywhere(
    tok: Token,
    *,
    axis: Axis,
    model_name: str = "all-MiniLM-L6-v2",
) -> bool:
    return _degree_score(tok, axis=axis, model_name=model_name) > 0.05


def _degree_strength(
    tok: Token,
    *,
    axis: Axis,
    model_name: str = "all-MiniLM-L6-v2",
) -> Strength:
    s = _degree_score(tok, axis=axis, model_name=model_name)
    if s >= 0.75:
        return Strength.STRONG
    if s >= 0.35:
        return Strength.MED
    return Strength.WEAK


def _invert_dir(d: Direction) -> Direction:
    return Direction.LOWER if d == Direction.RAISE else Direction.RAISE


# ---------------------------------------------------------------------
# Direction inference (semantic poles; no token tables)
# ---------------------------------------------------------------------


@lru_cache(maxsize=256)
def _axis_dir_poles(model_name: str) -> Dict[Axis, Tuple[str, str]]:
    """
    Returns (raise_pole_text, lower_pole_text) for direction inference.
    These are conceptual descriptions (not synonym lists).

    Contract:
    - raise_pole  => semantic meaning of "MORE / INCREASE"
    - lower_pole  => semantic meaning of "LESS / DECREASE"
    """
    return {
        Axis.BRIGHTNESS: (
            "higher lightness; lighter; brighter; more light reflected; closer to white",
            "lower lightness; darker; dimmer; closer to black",
        ),
        Axis.DEPTH: (
            "greater depth; deeper; darker; inkier; denser color mass",
            "less depth; shallower; lighter; less inky",
        ),
        Axis.SATURATION: (
            "higher chroma; more saturated; stronger color intensity; less muted",
            "lower chroma; less saturated; more muted; more greyed",
        ),
        Axis.VIBRANCY: (
            "stronger visual pop; more vibrant; more vivid; punchy; neon or electric appearance",
            "weaker visual pop; less vibrant; subdued; muted energy; not neon or flashy",
        ),
        Axis.CLARITY: (
            "cleaner and clearer appearance; crisp; sharp; not muddy or hazy",
            "less clear; muddier; hazier; softer or cloudy appearance",
        ),
    }

def _axis_direct_base_dir(lemma: str, *, axis: Axis) -> Optional[Direction]:
    l = (lemma or "").strip().lower()
    if not l:
        return None

    if axis == Axis.SATURATION:
        if l in {"saturated", "saturate"}:
            return Direction.RAISE
        if l in {"desaturated", "desaturate"}:
            return Direction.LOWER

    # optionnel si tu veux aussi sécuriser BRIGHTNESS:
    # if axis == Axis.BRIGHTNESS and l in {"bright"}: return Direction.RAISE
    # if axis == Axis.BRIGHTNESS and l in {"dim"}: return Direction.LOWER

    return None

def _semantic_direction(
    tok: Token,
    *,
    axis: Axis,
    model_name: str,
    min_delta: float = 0.03,
) -> Optional[Direction]:
    lemma = _norm_text_piece(tok.lemma_) or _norm_text_piece(tok.text)
    if not lemma:
        return None

    # ------------------------------------------------------------------
    # HARD DIRECTION ANCHOR FOR AXIS-DIRECT ADJECTIVES
    # (prevents embedding confusion like "saturated" ~ "less saturated")
    # ------------------------------------------------------------------
    l = lemma.lower()

    if axis == Axis.SATURATION:
        if l in {"saturated", "saturate"}:
            return Direction.RAISE
        if l in {"desaturated", "desaturate"}:
            return Direction.LOWER

    # (optionnel mais recommandé si tu veux bétonner aussi ces axes)
    # if axis == Axis.BRIGHTNESS:
    #     if l in {"bright"}:
    #         return Direction.RAISE
    #     if l in {"dim", "dark"}:
    #         return Direction.LOWER

    poles = _axis_dir_poles(model_name).get(axis)
    if not poles:
        return None

    raise_txt, lower_txt = poles

    v = _text_vec(lemma, model_name=model_name)
    v_raise = _text_vec(raise_txt, model_name=model_name)
    v_lower = _text_vec(lower_txt, model_name=model_name)

    s_raise = _cos_sim(v, v_raise)
    s_lower = _cos_sim(v, v_lower)

    if abs(float(s_raise) - float(s_lower)) < float(min_delta):
        return None

    return Direction.RAISE if s_raise > s_lower else Direction.LOWER


def _direction_from_context(tok: Token, *, axis: Axis, model_name: str) -> Direction:
    """
    Direction inference contract (FIXED):
      - base_dir from semantic poles; if None, fall back to axis-extremeness direction.
      - degree_op modifies intent:
          MORE -> keep base_dir
          LESS -> invert(base_dir)
          CAP  -> invert(base_dir)  ("too bright" => want lower brightness)
      - negation flips ONLY when op is NONE (plain "not bright", "not muddy").
        For "not too X", negation scopes over "too" and MUST NOT flip again.
    """
    base_dir = _semantic_direction(tok, axis=axis, model_name=model_name)
    if base_dir is None:
        _ext, ext_dir = _axis_extremeness(tok, axis=axis, model_name=model_name)
        base_dir = ext_dir

    op, _ = _degree_operator(tok, model_name=model_name)
    neg = _has_negation_anywhere(tok, model_name=model_name)

    if op == "MORE":
        return base_dir
    if op == "LESS":
        return _invert_dir(base_dir)
    if op == "CAP":
        # "too X" or "not too X" => avoid the implied extreme => invert base_dir
        return _invert_dir(base_dir)

    # NONE: plain negation flips
    return _invert_dir(base_dir) if neg else base_dir


def _evidence(tok: Token, *, model_name: str = "all-MiniLM-L6-v2") -> tuple[str, int, int]:
    """
    Evidence must be noise-free and reconstructable:
    - Keep only: negation cue(s) + grammatical degree operator(s) + head token.
    - Do NOT include discourse intensifiers (e.g. "really") => no advmod harvesting.
    """
    doc = tok.doc

    toks: List[Token] = [tok]

    # negation cues (syntactic + semantic window)
    toks.extend([ch for ch in tok.children if ch.dep_ == "neg"])
    toks.extend(_neg_cues_in_window(tok, window=3, model_name=model_name))

    # grammatical degree operators only (small fixed set: not a "noise list")
    def _add_degree_ops(root: Token) -> None:
        for t in root.subtree:
            lem = _norm_text_piece(t.lemma_) or _norm_text_piece(t.text)
            if lem in {"too", "more", "less"}:
                toks.append(t)

    _add_degree_ops(tok)
    head = getattr(tok, "head", None)
    if head is not None and head is not tok:
        _add_degree_ops(head)

    # also check left window for "too/more/less" when parse is weird
    for j in (tok.i - 1, tok.i - 2, tok.i - 3):
        if j >= 0:
            t = doc[j]
            lem = _norm_text_piece(t.lemma_) or _norm_text_piece(t.text)
            if lem in {"too", "more", "less"}:
                toks.append(t)

    toks = sorted({t.i: t for t in toks}.values(), key=lambda t: t.i)

    start = toks[0].idx
    last = toks[-1]
    end = last.idx + len(last.text)

    ev_parts = [_norm_text_piece(t.text) for t in toks]
    evidence = " ".join(" ".join(ev_parts).split()).strip().lower()

    return evidence, int(start), int(end)


# ---------------------------------------------------------------------
# Candidate selection fixes (STOP treating domain nouns like "lipstick" as constraints)
# ---------------------------------------------------------------------


def _redirect_to_amod_adj(tok: Token) -> Token:
    """
    If tok is a NOUN/PROPN and has adjectival modifiers (amod ADJ),
    we want to score/extract on the ADJ (e.g., "bright") not on the noun ("lipstick").
    """
    if tok.pos_ in {"NOUN", "PROPN"}:
        amods = [ch for ch in tok.children if ch.dep_ == "amod" and ch.pos_ == "ADJ"]
        if amods:
            return sorted(amods, key=lambda t: t.i)[-1]
    return tok


def _is_adjectival_noun(tok: Token) -> bool:
    if tok.pos_ != "NOUN":
        return False
    if tok.is_stop or tok.is_punct or tok.is_space:
        return False

    if _norm_text_piece(tok.lemma_) in _COLOR_DOMAIN_NOUNS:
        return False

    if len(tok.text) <= 2:
        return False

    if tok.dep_ in {"amod", "acomp", "attr"}:
        return True

    if any(ch.dep_ == "advmod" and ch.pos_ == "ADV" for ch in tok.children):
        return True

    return False


def _is_constraint_candidate(tok: Token) -> bool:
    if tok.is_stop or tok.is_punct or tok.is_space:
        return False
    if len(tok.text) <= 2:
        return False

    lem = _norm_text_piece(tok.lemma_) or _norm_text_piece(tok.text)
    head = getattr(tok, "head", None)
    if (
        head is not None
        and head.pos_ in {"NOUN", "PROPN"}
        and _norm_text_piece(head.lemma_) in _COLOR_DOMAIN_NOUNS
        and tok.dep_ in {"amod", "compound"}
    ):
        if lem in _AXIS_LEXICON:
            return True
        if lem.endswith("ness") and len(lem) > 4 and (lem[:-4] in _AXIS_LEXICON):
            return True
        if lem.endswith("ed") and len(lem) > 3 and (lem[:-2] in _AXIS_LEXICON):
            return True

    if tok.pos_ in {"VERB", "AUX"}:
        if tok.dep_ == "amod" and tok.tag_ in {"VBN", "VBD"}:
            return True
        return False

    if tok.pos_ == "ADJ":
        return True

    if _is_adjectival_noun(tok):
        return True

    if _has_negation(tok) and tok.pos_ in {"ADJ", "ADV", "NOUN"}:
        if tok.pos_ == "NOUN" and _norm_text_piece(tok.lemma_) in _COLOR_DOMAIN_NOUNS:
            return False
        return True

    if any(ch.dep_ == "advmod" and ch.pos_ == "ADV" for ch in tok.children):
        if tok.pos_ == "NOUN" and _norm_text_piece(tok.lemma_) in _COLOR_DOMAIN_NOUNS:
            return False
        return True

    return False


def _head_is_domain_noun(tok: Token) -> bool:
    head = getattr(tok, "head", None)
    if head is None:
        return False
    if head.pos_ not in {"NOUN", "PROPN"}:
        return False
    return _norm_text_piece(head.lemma_) in _COLOR_DOMAIN_NOUNS


def _quality_label(*, axis_gate: str, conf: float, margin: float) -> str:
    gate = (axis_gate or "").upper()
    if gate in {"STRICT", "LEXICON"} and conf >= 0.50 and margin >= 0.10:
        return "high"
    if gate in {"STRICT", "LEXICON"}:
        return "med"
    if conf >= 0.45 and margin >= 0.05:
        return "med"
    return "low"


def _strength_rank(s: Strength) -> int:
    if s == Strength.STRONG:
        return 3
    if s == Strength.MED:
        return 2
    if s == Strength.WEAK:
        return 1
    return 0


def _merge_constraints(constraints: List[Constraint]) -> List[Constraint]:
    buckets: Dict[Tuple[Axis, Direction, int], List[Constraint]] = {}
    for c in constraints:
        buckets.setdefault((c.axis, c.direction, int(c.clause_id)), []).append(c)

    out: List[Constraint] = []
    for _key, items in buckets.items():
        if len(items) == 1:
            c0 = items[0]
            meta = dict(c0.meta or {})
            meta.setdefault("evidence_parts", [c0.evidence] if c0.evidence else [])
            out.append(
                Constraint(
                    axis=c0.axis,
                    direction=c0.direction,
                    strength=c0.strength,
                    evidence=c0.evidence,
                    clause_id=c0.clause_id,
                    confidence=c0.confidence,
                    scope=c0.scope,
                    meta=meta,
                )
            )
            continue

        items_sorted = sorted(items, key=lambda c: (float(c.confidence), _strength_rank(c.strength)), reverse=True)
        best = items_sorted[0]

        evidences: List[str] = []
        for c in items:
            ev = (c.evidence or "").strip()
            if ev and ev not in evidences:
                evidences.append(ev)

        merged_evidence = " / ".join(evidences) if evidences else best.evidence
        merged_conf = max(float(c.confidence) for c in items)
        merged_strength = max((c.strength for c in items), key=_strength_rank)

        meta = dict(best.meta or {})
        meta["evidence_parts"] = evidences
        meta["merged_count"] = len(items)
        meta["confidence_merged_max"] = merged_conf

        out.append(
            Constraint(
                axis=best.axis,
                direction=best.direction,
                strength=merged_strength,
                evidence=merged_evidence,
                clause_id=best.clause_id,
                confidence=merged_conf,
                scope=best.scope,
                meta=meta,
            )
        )

    return out


def extract_constraints_from_doc(
    doc: Doc,
    *,
    clause_id: int,
    clause_polarity: Polarity = Polarity.UNKNOWN,
    blocked_lemmas: Optional[set[str]] = None,
    mapper_model: str = "all-MiniLM-L6-v2",
    mapper_threshold: float = 0.35,
    mapper_min_margin: float = 0.08,
) -> List[Constraint]:
    """Does: extract normalized constraints from a full document.
    Aggregates clause-level constraints into document-level signals.
    """
    out: List[Constraint] = []

    blocked = {(_norm_text_piece(x) or x) for x in (blocked_lemmas or set()) if (x or "").strip()}

    for tok in doc:
        if not _is_constraint_candidate(tok):
            continue

        tok = _redirect_to_amod_adj(tok)

        # block both lemma and surface (robust to spaCy lemmatization oddities)
        lem = _norm_text_piece(getattr(tok, "lemma_", "") or tok.text)
        surf = _norm_text_piece(tok.text)
        if blocked and ((lem in blocked) or (surf in blocked)):
            continue

        axis, axis_meta = _axis_from_token(
            tok,
            mapper_model=mapper_model,
            mapper_threshold=mapper_threshold,
            mapper_min_margin=mapper_min_margin,
        )
        if axis is None:
            continue

        axis_score = float(axis_meta.get("axis_score", 0.0))
        axis_margin = float(axis_meta.get("axis_margin", 0.0))
        axis_gate = str(axis_meta.get("axis_gate", "STRICT"))
        fam = str(axis_meta.get("axis_family") or _family(axis))

        has_signal = bool(
            _has_negation_anywhere(tok, model_name=mapper_model) or _has_degree_anywhere(tok, axis=axis, model_name=mapper_model)
        )
        head_domain = _head_is_domain_noun(tok)

        if axis_gate not in {"STRICT", "LEXICON"} and axis_margin < 0.03:
            if not (has_signal and fam == "LIGHTNESS"):
                continue

        if not has_signal:
            if axis_score < 0.40 and axis_margin < 0.02:
                continue

        if axis_gate not in {"STRICT", "LEXICON"}:
            if has_signal and fam == "LIGHTNESS":
                margin_floor = 0.02
                score_floor = 0.43
            elif has_signal and (fam == "CHROMA") and (not head_domain):
                margin_floor = 0.03
                score_floor = 0.45
            else:
                margin_floor = 0.05
                score_floor = 0.55
            if axis_margin < margin_floor or axis_score < score_floor:
                continue

        op, _op_sc = _degree_operator(tok, model_name=mapper_model)

        direction = _direction_from_context(tok, axis=axis, model_name=mapper_model)
        strength = _degree_strength(tok, axis=axis, model_name=mapper_model)

        if _has_negation_anywhere(tok, model_name=mapper_model) and strength == Strength.WEAK:
            strength = Strength.MED

        evidence, ev_start, ev_end = _evidence(tok, model_name=mapper_model)
        quality = _quality_label(axis_gate=axis_gate, conf=axis_score, margin=axis_margin)
        deg_score = float(_degree_score(tok, axis=axis, model_name=mapper_model))
        ax_ext, ax_dir = _axis_extremeness(tok, axis=axis, model_name=mapper_model)

        out.append(
            Constraint(
                axis=axis,
                direction=direction,
                strength=strength,
                evidence=evidence,
                clause_id=clause_id,
                confidence=axis_score,
                scope=None,
                meta={
                    **axis_meta,
                    "tok": tok.text,
                    "negated": _has_negation_anywhere(tok, model_name=mapper_model),
                    "children": [(c.text, c.dep_, c.pos_, c.lemma_) for c in tok.children],
                    "direction": direction.value,
                    "strength": strength.value,
                    "degree_score": deg_score,
                    "degree_op": op,
                    "axis_extremeness": float(ax_ext),
                    "axis_extremeness_dir": ax_dir.value,
                    # keep both names (some tests/loggers expect one or the other)
                    "evidence_char_start": ev_start,
                    "evidence_char_end": ev_end,
                    "evidence_global_start": ev_start,
                    "evidence_global_end": ev_end,
                    "clause_polarity": clause_polarity.value,
                    "has_signal": bool(has_signal),
                    "axis_family_effective": fam,
                    "head_is_domain_noun": bool(head_domain),
                    "quality": quality,
                },
            )
        )

    return _merge_constraints(out)


def extract_constraints_from_clause_text(
    clause_text: str,
    *,
    clause_id: int,
    clause_polarity: Polarity = Polarity.UNKNOWN,
    blocked_lemmas: Optional[set[str]] = None,
    nlp: Language,
    mapper_model: str = "all-MiniLM-L6-v2",
    mapper_threshold: float = 0.35,
    mapper_min_margin: float = 0.08,
) -> List[Constraint]:
    """Does: extract constraints from a single clause text span.
    Used by: document-level constraint aggregation.
    """
    doc = nlp(clause_text)
    return extract_constraints_from_doc(
        doc,
        clause_id=clause_id,
        clause_polarity=clause_polarity,
        blocked_lemmas=blocked_lemmas,
        mapper_model=mapper_model,
        mapper_threshold=mapper_threshold,
        mapper_min_margin=mapper_min_margin,
    )


def extract_constraints(
    clauses: Iterable[Tuple[int, str]],
    *,
    clause_polarities: Optional[Dict[int, Polarity]] = None,
    blocked_lemmas: Optional[set[str]] = None,
    nlp: Language,
    mapper_model: str = "all-MiniLM-L6-v2",
    mapper_threshold: float = 0.35,
    mapper_min_margin: float = 0.08,
) -> List[Constraint]:
    """Does: unified entry point for constraint extraction.
    Handles both document- and clause-level inputs.
    """
    out: List[Constraint] = []
    for cid, text in clauses:
        doc = nlp(text)
        pol = Polarity.UNKNOWN if clause_polarities is None else clause_polarities.get(cid, Polarity.UNKNOWN)
        out.extend(
            extract_constraints_from_doc(
                doc,
                clause_id=cid,
                clause_polarity=pol,
                blocked_lemmas=blocked_lemmas,
                mapper_model=mapper_model,
                mapper_threshold=mapper_threshold,
                mapper_min_margin=mapper_min_margin,
            )
        )
    return _merge_constraints(out)


__all__ = [
    "extract_constraints_from_doc",
    "extract_constraints_from_clause_text",
    "extract_constraints",
]
