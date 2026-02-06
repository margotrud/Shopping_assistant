# src/Shopping_assistant/nlp/llm/analyze_clauses.py
"""
LLM-based clause analysis.

Analyzes user text into structured semantic clauses
used to inform color intent and constraint resolution.
"""

from __future__ import annotations

import colorsys
import logging
import os
import re
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypedDict

import numpy as np

from Shopping_assistant.nlp.axes.predictor import predict_axis
from Shopping_assistant.nlp.parsing.clauses import ClauseChunk
from Shopping_assistant.nlp.parsing.polarity import PolarityLLM, infer_polarity_for_mentions
from Shopping_assistant.nlp.runtime.lexicon import ColorLexicon
from Shopping_assistant.nlp.runtime.spacy_runtime import load_spacy
from Shopping_assistant.utils.optional_deps import require

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hex -> Lab (local import to avoid circular imports via reco/__init__.py)
# ---------------------------------------------------------------------------


def _hex_to_lab_safe(hex_str: str):
    """Does: convert hex to Lab via reco._colorconv with a local import to avoid import-time cycles."""
    try:
        from Shopping_assistant.reco._colorconv import _hex_to_lab  # local import
    except Exception:
        return None
    try:
        return _hex_to_lab(hex_str)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public typed outputs
# ---------------------------------------------------------------------------


class LikeItem(TypedDict, total=False):
    """Does: represent a positively liked item inferred by the LLM."""

    family: str
    hue_deg: float
    alias: str
    intensity: Optional[str]


class DislikeItem(TypedDict, total=False):
    """Does: represent a negatively disliked item inferred by the LLM."""

    family: str
    hue_deg: float
    alias: str
    threshold: Optional[str]


class StyleIntent(TypedDict, total=False):
    """Does: represent a stylistic preference inferred from free text."""

    label: str
    strength: float
    polarity: str
    source: str


class Mention(TypedDict, total=False):
    """Does: represent a raw mention extracted by the LLM with minimal structure."""

    alias: str
    name: str
    hex: str
    lab_L: float
    lab_a: float
    lab_b: float
    hue_deg: float
    lab_hue_deg: float
    tok_start: int
    tok_len: int


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------


def _debug_enabled() -> bool:
    return os.environ.get("SA_COLOR_DEBUG", "").strip() in {"1", "true", "True", "YES", "yes"}


def _dbg(msg: str, **kv: Any) -> None:
    if not _debug_enabled():
        return
    parts = [msg]
    for k, v in kv.items():
        try:
            parts.append(f"{k}={v!r}")
        except Exception:
            parts.append(f"{k}=<unrepr>")
    logger.debug("[COLORDBG] %s", " ".join(parts))


def _pred_dbg(pred: Any) -> Dict[str, Any]:
    """
    Does:
        Normalize AxisPred debug fields across call sites.
        Uses confidence/margin + meta['source'] (never 'score'/'source' attributes).
    """
    try:
        meta = getattr(pred, "meta", None) or {}
    except Exception:
        meta = {}

    try:
        axis = getattr(pred, "axis", None)
    except Exception:
        axis = None

    try:
        conf = getattr(pred, "confidence", None)
    except Exception:
        conf = None

    try:
        margin = getattr(pred, "margin", None)
    except Exception:
        margin = None

    src = None
    if isinstance(meta, dict):
        src = meta.get("source")

    return {
        "axis": axis,
        "confidence": conf,
        "margin": margin,
        "source": src,
    }


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in s).split())


def _token_freq_over_aliases(aliases: Iterable[str]) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for a in aliases:
        for t in a.split():
            if t:
                freq[t] = freq.get(t, 0) + 1
    return freq


def _build_xkcd_single_token_allowset(*, css_aliases: set[str], xkcd_aliases: set[str]) -> set[str]:
    xkcd_multi = {a for a in xkcd_aliases if " " in a}
    css_multi = {a for a in css_aliases if " " in a}
    freq = _token_freq_over_aliases(xkcd_multi | css_multi)

    vals = sorted(freq.values())
    if not vals:
        return set()

    med = vals[len(vals) // 2]
    thr = max(3, int(med))

    allow: set[str] = set()
    for a in xkcd_aliases:
        if " " in a:
            continue
        if not a or len(a) < 3:
            continue
        if a in css_aliases:
            allow.add(a)
            continue
        if freq.get(a, 0) >= thr:
            allow.add(a)
    return allow


# ---------------------------------------------------------------------------
# Hex -> hue / Lab helpers
# ---------------------------------------------------------------------------


def _hex_to_rgb01(hx: str) -> Optional[Tuple[float, float, float]]:
    if not isinstance(hx, str):
        return None
    s = hx.strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        return (r, g, b)
    except Exception:
        return None


def _hex_to_hue_deg(hx: str) -> Optional[float]:
    rgb01 = _hex_to_rgb01(hx)
    if rgb01 is None:
        return None
    h, _s, _v = colorsys.rgb_to_hsv(*rgb01)
    return float((h * 360.0) % 360.0)


def _srgb01_to_linear(x: float) -> float:
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def _lab_hue_deg_from_ab(a: float, b: float) -> float:
    return float(np.degrees(np.arctan2(b, a)) % 360.0)


# ---------------------------------------------------------------------------
# World alias index (CSS + XKCD)
# ---------------------------------------------------------------------------


def _iter_css_color_names() -> Iterable[str]:
    webcolors = require("webcolors", extra="webcolors", purpose="CSS colors name inventory")
    if hasattr(webcolors, "names"):
        try:
            return list(webcolors.names("css3"))
        except Exception:
            return []
    if hasattr(webcolors, "CSS3_NAMES_TO_HEX"):
        return list(getattr(webcolors, "CSS3_NAMES_TO_HEX").keys())
    return []


def _iter_xkcd_color_items() -> Iterable[Tuple[str, str]]:
    require("matplotlib", extra="matplotlib", purpose="XKCD colors name inventory")
    from matplotlib import colors as mcolors  # type: ignore

    return list(getattr(mcolors, "XKCD_COLORS").items())


def _norm_alias(s: str) -> str:
    s = s.lower().strip()
    if s.startswith("xkcd:"):
        s = s[5:]
    out: List[str] = []
    prev_space = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_space = False
        else:
            if not prev_space:
                out.append(" ")
                prev_space = True
    return " ".join("".join(out).split())


@lru_cache(maxsize=2)
def build_world_alias_index(*, include_xkcd: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Does:
        Build and cache a normalized alias->colors-info index for CSS (+ optional XKCD).
        Cached across calls to eliminate first-call 10s+ rebuilds in interpret_nlp().
    """
    webcolors = require("webcolors", extra="webcolors", purpose="CSS colors inventory")
    idx: Dict[str, Dict[str, Any]] = {}

    css_aliases: set[str] = set()
    for name in _iter_css_color_names():
        alias = _norm_alias(name)
        if not alias:
            continue
        css_aliases.add(alias)

        try:
            hx = webcolors.name_to_hex(name)
        except Exception:
            hx = None

        info: Dict[str, Any] = {"name": alias, "source": "css"}
        if hx:
            info["hex"] = hx
            hue = _hex_to_hue_deg(hx)
            if hue is not None:
                info["hue_deg"] = hue
            lab = _hex_to_lab_safe(hx)
            if lab is not None:
                L, a, b = lab
                info["lab_L"] = float(L)
                info["lab_a"] = float(a)
                info["lab_b"] = float(b)
                info["lab_hue_deg"] = _lab_hue_deg_from_ab(a, b)

        idx[alias] = info

    if include_xkcd:
        xkcd_items = list(_iter_xkcd_color_items())
        xkcd_aliases: set[str] = set()

        for raw_name, _hx in xkcd_items:
            alias = _norm_alias(raw_name)
            if alias:
                xkcd_aliases.add(alias)

        xkcd_single_allow = _build_xkcd_single_token_allowset(
            css_aliases=css_aliases,
            xkcd_aliases=xkcd_aliases,
        )

        for raw_name, hx in xkcd_items:
            alias = _norm_alias(raw_name)
            if not alias:
                continue

            if " " not in alias and alias not in xkcd_single_allow:
                continue

            info2: Dict[str, Any] = {"name": alias, "hex": hx, "source": "xkcd"}

            hue2 = _hex_to_hue_deg(hx)
            if hue2 is not None:
                info2["hue_deg"] = hue2

            lab2 = _hex_to_lab_safe(hx)
            if lab2 is not None:
                L2, a2, b2 = lab2
                info2["lab_L"] = float(L2)
                info2["lab_a"] = float(a2)
                info2["lab_b"] = float(b2)
                info2["lab_hue_deg"] = _lab_hue_deg_from_ab(a2, b2)

            idx[alias] = info2

    return idx


def _max_ngram_from_index(color_index: Dict[str, Dict[str, Any]]) -> int:
    if not color_index:
        return 1
    return min(max(len(a.split()) for a in color_index.keys()), 4)


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lexicon resolve (controlled semantic)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _color_lexicon() -> Optional[ColorLexicon]:
    try:
        from Shopping_assistant.nlp.runtime.lexicon import load_default_lexicon

        return load_default_lexicon()
    except Exception as e:
        _dbg("lexicon_load_failed", err=str(e))
        return None


def _lexicon_resolve(cand: str, *, allow_semantic: bool) -> Optional[Dict[str, Any]]:
    lex = _color_lexicon()
    if lex is None:
        return None

    resolved = lex.resolve(
        cand,
        topk=1,
        fuzzy_cutoff=75.0,
        use_semantic=bool(allow_semantic),
    )
    if not resolved:
        _dbg("lexicon_no_match", cand=cand, allow_semantic=allow_semantic)
        return None

    best = resolved[0]
    hx = best.hex
    src = f"lexicon:{best.source}"

    info: Dict[str, Any] = {"name": best.name, "hex": hx, "source": src}

    hue = _hex_to_hue_deg(hx)
    if hue is not None:
        info["hue_deg"] = hue

    lab = _hex_to_lab_safe(hx)
    if lab is not None:
        L, a, b = lab
        info["lab_L"] = float(L)
        info["lab_a"] = float(a)
        info["lab_b"] = float(b)
        info["lab_hue_deg"] = _lab_hue_deg_from_ab(a, b)

    _dbg(
        "lexicon_match",
        cand=cand,
        allow_semantic=allow_semantic,
        best_alias=getattr(best, "alias", None),
        best_name=best.name,
        best_source=best.source,
        best_score=best.score,
    )
    return info


# ---------------------------------------------------------------------------
# spaCy
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _stop_words_en() -> set[str]:
    spacy = require("spacy", extra="spacy", purpose="STOP_WORDS for mention/style extraction.")
    try:
        nlp = load_spacy("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
    return set(getattr(nlp.Defaults, "stop_words", set()))


@lru_cache(maxsize=1)
def _spacy_nlp_en():
    spacy = require("spacy", extra="spacy", purpose="spaCy for mention extraction.")
    try:
        return load_spacy("en_core_web_sm")
    except Exception:
        return spacy.blank("en")


# ---------------------------------------------------------------------------
# Product filtering (style extraction)
# ---------------------------------------------------------------------------

PRODUCT_BASES: set[str] = {"lipstick"}

# Keep minimal (domain bootstrap only): allow "nude" via env-provided hex.
NUDE_HEX = os.environ.get("SA_COLOR_NUDE_HEX", "#e6c2b3").strip().lower()


def _product_base_form(token: str) -> str:
    t = token.strip().lower()
    if not t:
        return t
    if t in PRODUCT_BASES:
        return t
    if t.endswith("es") and t[:-2] in PRODUCT_BASES:
        return t[:-2]
    if t.endswith("s") and t[:-1] in PRODUCT_BASES:
        return t[:-1]
    return t


def _nude_info_if_applicable(cand: str) -> Optional[Dict[str, Any]]:
    if _norm(cand) != "nude":
        return None
    hx = NUDE_HEX if NUDE_HEX.startswith("#") else f"#{NUDE_HEX}"
    if not re.fullmatch(r"^#[0-9a-f]{6}$", hx):
        hx = "#e6c2b3"
    return {"name": "Nude", "hex": hx, "source": "hardcode:nude"}


# ---------------------------------------------------------------------------
# Mention extraction
# ---------------------------------------------------------------------------


def extract_mentions_free(
    doc_text: str,
    color_index: Dict[str, Dict[str, Any]],
    doc: Optional[Any] = None,
) -> List[Mention]:
    """Does: extract free-form mentions using an LLM prompt.
    Returns: list of mention-like dicts or objects for downstream parsing.
    """
    if not doc_text:
        return []

    stop_words = _stop_words_en()

    # Fallback (safe but conservative)
    if doc is None:
        toks = _norm(doc_text).split()
        if not toks:
            return []

        max_n = _max_ngram_from_index(color_index)
        hits: List[Mention] = []
        i = 0

        while i < len(toks):
            matched: Optional[Mention] = None
            for n in range(min(max_n, len(toks) - i), 0, -1):
                span = toks[i : i + n]

                if span and span[0] == "not" and len(span) >= 2:
                    # keep existing behavior: "not X" -> "X"
                    cand_span = [span[-1]]
                    trimmed_left = len(span) - 1
                else:
                    cand_span = span
                    orig_len = len(cand_span)
                    while cand_span and cand_span[0] in stop_words:
                        cand_span = cand_span[1:]
                    trimmed_left = orig_len - len(cand_span)

                    if not cand_span:
                        continue

                cand_len = len(cand_span)
                cand = " ".join(cand_span)

                head = cand.split()[-1]
                if head in stop_words:
                    continue

                info = None
                src = None
                if " " not in cand:
                    nude_info = _nude_info_if_applicable(head)
                    if nude_info:
                        info = nude_info
                        src = "hardcode"

                if _product_base_form(head) in PRODUCT_BASES:
                    continue

                if " " not in cand:
                    pred0 = predict_axis(head, debug=False)
                    if pred0.axis is not None:
                        continue

                # Lookup in color_index FIRST.
                if info is None:
                    info = color_index.get(cand)
                    src = "world_exact" if info else None

                if not info and " " not in cand:
                    if not (head.endswith("y") or head.endswith("ish") or head == "nude"):
                        continue

                if info:
                    _dbg("world_exact_hit", cand=cand, name=info.get("name"), hx=info.get("hex"))

                if not info:
                    allow_sem = (" " in cand)
                    _dbg("call_lexicon", cand=cand, allow_semantic=allow_sem)
                    info = _lexicon_resolve(cand, allow_semantic=allow_sem)
                    src = "lexicon" if info else None

                if not info:
                    continue

                matched = {
                    "alias": cand if src != "hardcode" else head,
                    "name": str(info.get("name") or cand),
                    "tok_start": i + trimmed_left,
                    "tok_len": cand_len if src != "hardcode" else 1,
                }

                hx = info.get("hex")
                if isinstance(hx, str) and hx:
                    matched["hex"] = hx

                L = _safe_float(info.get("lab_L"))
                a = _safe_float(info.get("lab_a"))
                b = _safe_float(info.get("lab_b"))
                if L is not None and a is not None and b is not None:
                    matched["lab_L"] = L
                    matched["lab_a"] = a
                    matched["lab_b"] = b

                lab_h = _safe_float(info.get("lab_hue_deg"))
                if lab_h is not None:
                    matched["lab_hue_deg"] = lab_h
                else:
                    hue = _safe_float(info.get("hue_deg"))
                    if hue is not None:
                        matched["hue_deg"] = hue

                _dbg(
                    "fallback_match",
                    span=span,
                    cand=cand,
                    src=src,
                    name=matched["name"],
                    hx=matched.get("hex"),
                )
                i += n
                break

            if matched:
                hits.append(matched)
            else:
                i += 1

        seen: set[str] = set()
        uniq: List[Mention] = []
        for h in hits:
            nm = str(h.get("name") or "")
            if nm and nm not in seen:
                uniq.append(h)
                seen.add(nm)
        return uniq

    # spaCy-driven extraction
    FUNCTIONAL_POS = {
        "VERB",
        "AUX",
        "DET",
        "ADP",
        "PRON",
        "PART",
        "SCONJ",
        "CCONJ",
        "PUNCT",
        "SPACE",
    }
    COLORLIKE_POS = {"ADJ", "NOUN", "PROPN"}

    max_n = _max_ngram_from_index(color_index)
    hits: List[Mention] = []
    i = 0

    seen_cand_counts: Dict[str, int] = {}

    def _span_tokens(i0: int, n: int) -> List[Any]:
        return [doc[j] for j in range(i0, min(len(doc), i0 + n))]

    def _trim_left(ts: List[Any]) -> List[Any]:
        k = 0
        while k < len(ts):
            t = ts[k]
            if t.is_space or t.is_punct or t.is_stop or t.pos_ in {"DET", "ADP", "PRON", "PART"}:
                k += 1
                continue
            break
        return ts[k:] if k > 0 else ts

    def _trim_right(ts: List[Any]) -> List[Any]:
        k = len(ts) - 1
        while k >= 0:
            t = ts[k]
            if t.is_space or t.is_punct:
                k -= 1
                continue
            break
        return ts[: k + 1]

    def _head_color_token(ts: List[Any]) -> Optional[Any]:
        for t in reversed(ts):
            if t.is_space or t.is_punct:
                continue
            if not t.is_alpha:
                continue
            return t
        return None

    def _span_is_eligible(ts: List[Any]) -> bool:
        if not ts:
            return False

        head = _head_color_token(ts)
        if head is None:
            return False
        if head.is_stop:
            return False
        if head.pos_ in FUNCTIONAL_POS:
            return False
        if head.pos_ not in COLORLIKE_POS:
            return False
        return True

    def _span_has_coordination(ts: List[Any]) -> bool:
        for t in ts:
            if getattr(t, "pos_", None) == "CCONJ":
                return True
            if getattr(t, "dep_", None) in {"cc", "conj"}:
                return True
        return False

    def _has_axis_like_prefix(ts: List[Any]) -> bool:
        head = _head_color_token(ts)
        if head is None:
            return False

        for t in ts:
            if t.i == head.i:
                continue
            if t.is_stop or not t.is_alpha:
                continue

            t_txt = _norm(getattr(t, "text", "") or "")
            if t_txt and t_txt in color_index:
                continue

            pred = predict_axis(str(getattr(t, "lemma_", None) or t.text), debug=False)
            if pred.axis is not None:
                return True
        return False

    def _finalize_match(
        *,
        alias: str,
        info: Dict[str, Any],
        tok_start: int,
        tok_len: int,
        src: str,
        span_tokens: List[Any],
        head_tok: Any,
        allow_semantic: bool,
    ) -> Mention:
        matched: Mention = {
            "alias": alias,
            "name": str(info.get("name") or alias),
            "tok_start": int(tok_start),
            "tok_len": int(tok_len),
        }

        hx = info.get("hex")
        if isinstance(hx, str) and hx:
            matched["hex"] = hx

        L = _safe_float(info.get("lab_L"))
        a = _safe_float(info.get("lab_a"))
        b = _safe_float(info.get("lab_b"))
        if L is not None and a is not None and b is not None:
            matched["lab_L"] = L
            matched["lab_a"] = a
            matched["lab_b"] = b

        lab_h = _safe_float(info.get("lab_hue_deg"))
        if lab_h is not None:
            matched["lab_hue_deg"] = lab_h
        else:
            hue = _safe_float(info.get("hue_deg"))
            if hue is not None:
                matched["hue_deg"] = hue

        _dbg(
            "spacy_match",
            cand=matched["alias"],
            name=matched["name"],
            hx=matched.get("hex"),
            src=src,
            span=[t.text for t in span_tokens],
            head=getattr(head_tok, "text", None),
            allow_semantic=allow_semantic,
        )
        return matched

    while i < len(doc):
        ti = doc[i]
        if ti.is_space or ti.is_punct:
            i += 1
            continue

        matched: Optional[Mention] = None

        for n in range(min(max_n, len(doc) - i), 0, -1):
            ts0 = _span_tokens(i, n)
            ts = _trim_left(ts0)
            ts = _trim_right(ts)
            if not ts:
                continue
            if not _span_is_eligible(ts):
                continue

            head = _head_color_token(ts)
            if head is None:
                continue

            if ts and _norm(ts[0].text) == "not":
                cand = _norm(head.text)
            else:
                cand = _norm(" ".join(t.text for t in ts)).strip()

            if not cand:
                continue

            head_txt = _norm(head.text)

            if _debug_enabled():
                key = f"{cand}|head={head.text}|pos={head.pos_}|i={int(ts[0].i)}|n={n}"
                seen_cand_counts[key] = seen_cand_counts.get(key, 0) + 1
                _dbg(
                    "try_span",
                    cand=cand,
                    head=head.text,
                    head_lemma=getattr(head, "lemma_", None),
                    head_pos=getattr(head, "pos_", None),
                    head_is_stop=bool(getattr(head, "is_stop", False)),
                    span=[t.text for t in ts],
                    span_pos=[getattr(t, "pos_", None) for t in ts],
                    span_i=[int(t.i) for t in ts],
                    count=seen_cand_counts[key],
                )

            if head_txt in stop_words:
                _dbg("skip_stopword_head", cand=cand, head=head_txt)
                continue

            if " " not in cand:
                nude_info = _nude_info_if_applicable(head_txt)
                if nude_info:
                    matched = _finalize_match(
                        alias="nude",
                        info=nude_info,
                        tok_start=int(head.i),
                        tok_len=1,
                        src="hardcode",
                        span_tokens=ts,
                        head_tok=head,
                        allow_semantic=False,
                    )
                    i = matched["tok_start"] + matched["tok_len"]
                    break

            if _product_base_form(head_txt) in PRODUCT_BASES:
                if _debug_enabled():
                    _dbg(
                        "product_head_debug",
                        head=head.text,
                        head_i=int(head.i),
                        head_dep=getattr(head, "dep_", None),
                        head_children=[
                            (c.text, getattr(c, "dep_", None), getattr(c, "pos_", None), int(c.i))
                            for c in head.children
                        ],
                        span=[(t.text, getattr(t, "dep_", None), getattr(t, "pos_", None), int(t.i)) for t in ts],
                    )

                mods = [
                    t
                    for t in ts
                    if getattr(t, "dep_", "") in {"amod", "compound"} and t.is_alpha and not t.is_stop
                ]
                _dbg(
                    "product_mods_extracted",
                    head=head.text,
                    mods=[(m.text, getattr(m, "dep_", None), getattr(m, "pos_", None)) for m in mods],
                )

                if not mods:
                    _dbg("skip_product_no_mods", cand=cand, head=head_txt, span=[t.text for t in ts])
                    continue

                found = None
                for m in reversed(mods):
                    m_cand = _norm(m.text)
                    if not m_cand:
                        continue

                    nude_info = _nude_info_if_applicable(m_cand)
                    if nude_info:
                        found = (m, "nude", nude_info, "hardcode_mod")
                        break

                    info_m = color_index.get(m_cand)
                    if info_m:
                        found = (m, m_cand, info_m, "world_mod")
                        break

                    predm = predict_axis(m_cand)
                    if predm.axis is not None:
                        continue

                    src_m = "world_mod" if info_m else None
                    if info_m:
                        _dbg("world_exact_hit_mod", cand=m_cand, name=info_m.get("name"), hx=info_m.get("hex"))

                    if not info_m:
                        _dbg("call_lexicon", cand=m_cand, allow_sem=False)
                        info_m = _lexicon_resolve(m_cand, allow_semantic=False)
                        src_m = "lexicon_mod" if info_m else None

                    if info_m:
                        found = (m, m_cand, info_m, src_m or "mod")
                        break

                if not found:
                    _dbg("skip_product_no_color_mod", cand=cand, head=head_txt, mods=[m.text for m in mods])
                    continue

                m_tok, m_cand, info, src = found
                matched = _finalize_match(
                    alias=m_cand,
                    info=info,
                    tok_start=int(m_tok.i),
                    tok_len=1,
                    src=src,
                    span_tokens=ts,
                    head_tok=head,
                    allow_semantic=False,
                )
                i = matched["tok_start"] + matched["tok_len"]
                break

            info = None
            src = None

            info = color_index.get(cand)
            src = "world_exact" if info else None
            if info:
                _dbg("world_exact_hit", cand=cand, name=info.get("name"), hx=info.get("hex"))

            if not info and any(getattr(t, "pos_", None) in {"VERB", "AUX"} for t in ts):
                _dbg("skip_verb_span_no_world", cand=cand, span=[t.text for t in ts])
                continue

            if not info and " " not in cand:
                pred0 = predict_axis(head_txt, debug=False)
                pd0 = _pred_dbg(pred0)
                _dbg(
                    "axis_pred_single",
                    cand=cand,
                    head=head_txt,
                    pos=getattr(head, "pos_", None),
                    axis=pd0["axis"],
                    confidence=pd0["confidence"],
                    margin=pd0["margin"],
                    source=pd0["source"],
                )
                if pred0.axis is not None:
                    _dbg("skip_axis_single", cand=cand, head=head_txt, axis=pred0.axis)
                    continue

            if not info and " " not in cand:
                if head.pos_ in {"ADJ", "ADV"} and not (cand.endswith("y") or cand.endswith("ish")):
                    _dbg("skip_generic_adj_single", cand=cand, head=head.text, pos=head.pos_)
                    continue

            if not info:
                allow_sem = (" " in cand)
                _dbg("call_lexicon", cand=cand, allow_sem=allow_sem)
                info = _lexicon_resolve(cand, allow_semantic=allow_sem)
                src = "lexicon" if info else None

            if not info:
                continue

            if len(ts) >= 2 and _has_axis_like_prefix(ts) and not _span_has_coordination(ts):
                cand_head = _norm(head.text)
                info2 = color_index.get(cand_head)
                src2 = "world_head" if info2 else None
                if info2:
                    _dbg("world_exact_hit_head", cand=cand_head, name=info2.get("name"), hx=info2.get("hex"))

                if not info2:
                    _dbg("call_lexicon", cand=cand_head, allow_sem=False)
                    info2 = _lexicon_resolve(cand_head, allow_semantic=False)
                    src2 = "lexicon_head" if info2 else None
                if not info2:
                    continue

                info = info2
                src = src2 or "head"
                matched = _finalize_match(
                    alias=cand_head,
                    info=info,
                    tok_start=int(head.i),
                    tok_len=1,
                    src=src,
                    span_tokens=ts,
                    head_tok=head,
                    allow_semantic=False,
                )
            else:
                matched = _finalize_match(
                    alias=cand,
                    info=info,
                    tok_start=int(ts[0].i),
                    tok_len=int(ts[-1].i - ts[0].i + 1),
                    src=src or "unk",
                    span_tokens=ts,
                    head_tok=head,
                    allow_semantic=(" " in cand),
                )

            i = matched["tok_start"] + matched["tok_len"]
            break

        if matched:
            hits.append(matched)
        else:
            i += 1

    seen: set[str] = set()
    uniq: List[Mention] = []
    for h in hits:
        nm = str(h.get("name") or "")
        if nm and nm not in seen:
            uniq.append(h)
            seen.add(nm)

    if _debug_enabled() and seen_cand_counts:
        top = sorted(seen_cand_counts.items(), key=lambda kv: -kv[1])[:30]
        _dbg("summary_top_attempts", top=top)

    return uniq


# ---------------------------------------------------------------------------
# Style extraction (unchanged)
# ---------------------------------------------------------------------------


def _extract_style_intents_from_chunks(
    chunks: List[ClauseChunk],
    color_index: Dict[str, Dict[str, Any]],
) -> List[StyleIntent]:
    stop_words = _stop_words_en()
    color_tokens = {tok for alias in color_index for tok in alias.split()}
    counts: Dict[str, int] = {}

    for ch in chunks:
        for t in _norm(ch["text"]).split():
            if (
                not t
                or len(t) < 3
                or t.isdigit()
                or t in color_tokens
                or t in stop_words
                or _product_base_form(t) in PRODUCT_BASES
            ):
                continue
            counts[t] = counts.get(t, 0) + 1

    counts = {k: v for k, v in counts.items() if v >= 2}
    if not counts:
        return []

    mx = max(counts.values())
    return [StyleIntent(label=k, strength=v / mx, polarity="LIKE", source="GLOBAL") for k, v in counts.items()]


def _extract_elliptical_style_intents(chunks: List[ClauseChunk]) -> List[StyleIntent]:
    stop_words = _stop_words_en()
    counts: Dict[str, int] = {}

    for ch in chunks:
        if ch.get("kind") != "ELLIPTICAL_NEG":
            continue
        for t in _norm(ch["text"]).split():
            if (
                not t
                or len(t) < 3
                or t.isdigit()
                or t in stop_words
                or _product_base_form(t) in PRODUCT_BASES
            ):
                continue
            counts[t] = counts.get(t, 0) + 1

    counts = {k: v for k, v in counts.items() if v >= 2}
    if not counts:
        return []

    mx = max(counts.values())
    return [StyleIntent(label=k, strength=v / mx, polarity="AVOID", source="ELLIPTICAL_NEG") for k, v in counts.items()]


def _extract_elliptical_descriptor_labels(chunks: List[ClauseChunk]) -> Dict[str, Any]:
    stop_words = _stop_words_en()
    labels: List[str] = []

    for ch in chunks:
        if ch.get("kind") != "ELLIPTICAL_NEG":
            continue
        for t in _norm(ch["text"]).split():
            if (
                not t
                or len(t) < 3
                or t.isdigit()
                or t in stop_words
                or _product_base_form(t) in PRODUCT_BASES
            ):
                continue
            labels.append(t)

    uniq = list(dict.fromkeys(labels))
    return {} if not uniq else {"RAW_LABELS": uniq}


_NOT_START_RE = re.compile(
    r"(?:(?P<sep>[,;:\(\)\[\]\-–—])\s*|\bbut\s+)\bnot\b\s+",
    flags=re.IGNORECASE,
)


def _find_elliptical_split(text: str) -> Optional[int]:
    m = _NOT_START_RE.search(text)
    if not m:
        return None
    inner = re.search(r"\bnot\b", text[m.start() : m.end()], flags=re.IGNORECASE)
    return None if not inner else m.start() + inner.start()


def analyze_clauses_with_llm(
    chunks: List[ClauseChunk],
    color_index: Dict[str, Dict[str, Any]],
    *,
    llm_polarity_fn: PolarityLLM,
    infer_clause_sentiment_fn: Optional[Callable[[str], Optional[str]]] = None,
    expose_styles: bool = True,
    debug_mentions: bool = False,
) -> Dict[str, Any]:
    """Does: analyze clauses with an LLM to extract mentions and intents.
    Used when rule-based parsing is insufficient or ambiguous.
    """
    if llm_polarity_fn is None:
        raise RuntimeError("analyze_clauses_with_llm requires llm_polarity_fn.")

    nlp = _spacy_nlp_en()
    if debug_mentions:
        os.environ["SA_COLOR_DEBUG"] = "1"

    likes: List[LikeItem] = []
    dislikes: List[DislikeItem] = []

    for ch in chunks:
        doc = nlp(ch["text"])
        mentions = extract_mentions_free(ch["text"], color_index, doc=doc)
        if not mentions:
            continue

        pols = infer_polarity_for_mentions(
            ch["text"],
            [m["name"] for m in mentions if "name" in m],
            llm_polarity_fn=llm_polarity_fn,
        )

        for m in mentions:
            nm = m.get("name")
            if not nm:
                continue
            hue = m.get("lab_hue_deg", m.get("hue_deg"))
            if pols.get(nm) == "LIKE":
                it: LikeItem = {"family": nm}
                if hue is not None:
                    it["hue_deg"] = float(hue)
                if m.get("alias"):
                    it["alias"] = str(m["alias"])
                likes.append(it)
            elif pols.get(nm) == "DISLIKE":
                it2: DislikeItem = {"family": nm}
                if hue is not None:
                    it2["hue_deg"] = float(hue)
                if m.get("alias"):
                    it2["alias"] = str(m["alias"])
                dislikes.append(it2)

    out: Dict[str, Any] = {"LIKES": likes, "DISLIKES": dislikes}

    if expose_styles:
        styles = _extract_style_intents_from_chunks(chunks, color_index)
        ellip = _extract_elliptical_style_intents(chunks)
        if styles or ellip:
            out["STYLE_INTENTS"] = styles + ellip

    desc = _extract_elliptical_descriptor_labels(chunks)
    if desc:
        out["CONSTRAINTS"] = desc

    _ = infer_clause_sentiment_fn
    return out


__all__ = [
    "LikeItem",
    "DislikeItem",
    "StyleIntent",
    "Mention",
    "build_world_alias_index",
    "extract_mentions_free",
    "analyze_clauses_with_llm",
    "_spacy_nlp_en",
]
