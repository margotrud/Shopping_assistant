# src/Shopping_assistant/nlp/llm/analyze_clauses.py
from __future__ import annotations

import colorsys
import re
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypedDict

import numpy as np

from Shopping_assistant.color.lexicon import ColorLexicon
from Shopping_assistant.nlp.parsing.clauses import ClauseChunk
from Shopping_assistant.nlp.parsing.polarity import PolarityLLM, infer_polarity_for_mentions
from Shopping_assistant.utils.optional_deps import require


# ---------------------------------------------------------------------------
# Public typed outputs
# ---------------------------------------------------------------------------

class LikeItem(TypedDict, total=False):
    family: str
    hue_deg: float
    alias: str
    intensity: Optional[str]


class DislikeItem(TypedDict, total=False):
    family: str
    hue_deg: float
    alias: str
    threshold: Optional[str]


class StyleIntent(TypedDict, total=False):
    label: str
    strength: float
    polarity: str
    source: str


class Mention(TypedDict, total=False):
    alias: str
    name: str
    hue_deg: float
    lab_hue_deg: float
    tok_start: int
    tok_len: int


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """
    Does:
        Normalize a string into lowercase alnum tokens separated by single spaces.
    """
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in s).split())


def _token_freq_over_aliases(aliases: Iterable[str]) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for a in aliases:
        toks = a.split()
        for t in toks:
            if not t:
                continue
            freq[t] = freq.get(t, 0) + 1
    return freq


def _build_xkcd_single_token_allowset(
    *,
    css_aliases: set[str],
    xkcd_aliases: set[str],
) -> set[str]:
    """
    Does:
        Build a dynamic allowlist for single-token XKCD aliases.
    """
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
        t = a
        if not t or len(t) < 3:
            continue
        if t in css_aliases:
            allow.add(t)
            continue
        if freq.get(t, 0) >= thr:
            allow.add(t)

    return allow


# ---------------------------------------------------------------------------
# Hex -> hue helpers (dynamic, derived from inventories)
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


def _hex_to_lab(hx: str) -> Optional[Tuple[float, float, float]]:
    rgb01 = _hex_to_rgb01(hx)
    if rgb01 is None:
        return None

    r, g, b = rgb01
    r, g, b = _srgb01_to_linear(r), _srgb01_to_linear(g), _srgb01_to_linear(b)

    # sRGB D65
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # D65 white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = X / Xn, Y / Yn, Z / Zn

    d = 6 / 29

    def f(t: float) -> float:
        return t ** (1 / 3) if t > d**3 else (t / (3 * d**2) + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b2 = 200 * (fy - fz)

    return (float(L), float(a), float(b2))


def _lab_hue_deg_from_ab(a: float, b: float) -> float:
    return float(np.degrees(np.arctan2(b, a)) % 360.0)


# ---------------------------------------------------------------------------
# World alias index (CSS + XKCD), dependency-driven
# ---------------------------------------------------------------------------

def _iter_css_color_names() -> Iterable[str]:
    webcolors = require("webcolors", extra="webcolors", purpose="CSS color name inventory")
    if hasattr(webcolors, "names"):
        try:
            return list(webcolors.names("css3"))
        except Exception:
            return []
    if hasattr(webcolors, "CSS3_NAMES_TO_HEX"):
        return list(getattr(webcolors, "CSS3_NAMES_TO_HEX").keys())
    return []


def _iter_xkcd_color_items() -> Iterable[Tuple[str, str]]:
    require("matplotlib", extra="matplotlib", purpose="XKCD color name inventory")
    from matplotlib import colors as mcolors  # type: ignore
    return list(getattr(mcolors, "XKCD_COLORS").items())


def _norm_alias(s: str) -> str:
    """
    Does:
        Normalize aliases (handles "xkcd:" prefix), keep alnum+space.
    """
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
        Build a dynamic alias index from CSS/XKCD color inventories.

    Returns:
        Mapping alias -> {name, hex, hue_deg, lab_hue_deg, source, ...}
    """
    webcolors = require("webcolors", extra="webcolors", purpose="CSS color inventory")
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

            lab = _hex_to_lab(hx)
            if lab is not None:
                _L, a, b = lab
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

            if " " not in alias:
                if alias not in xkcd_single_allow:
                    continue

            info2: Dict[str, Any] = {"name": alias, "hex": hx, "source": "xkcd"}

            hue2 = _hex_to_hue_deg(hx)
            if hue2 is not None:
                info2["hue_deg"] = hue2

            lab2 = _hex_to_lab(hx)
            if lab2 is not None:
                _L2, a2, b2 = lab2
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
# Extended lexicon (offline artifact) loader + resolver
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _color_lexicon() -> Optional[ColorLexicon]:
    """
    Does:
        Load the offline-built extended color lexicon if available.

    Notes:
        - No CSV dependency.
        - If missing/unavailable, return None and keep CSS/XKCD-only behavior.
    """
    try:
        return ColorLexicon.load()  # data/nlp/color_lexicon.json by default
    except Exception:
        return None


def _lexicon_info_for_candidate(cand: str) -> Optional[Dict[str, Any]]:
    """
    Does:
        Resolve a candidate phrase via ColorLexicon and convert to world-index-like info.

    Returns:
        {name, hex, hue_deg?, lab_hue_deg?, source}
    """
    lex = _color_lexicon()
    if lex is None:
        return None

    # Conservative: avoid spurious matches.
    resolved = lex.resolve(cand, topk=1, fuzzy_cutoff=75.0, use_semantic=True)
    if not resolved:
        return None

    best = resolved[0]
    hx = best.hex

    info: Dict[str, Any] = {
        "name": best.name,
        "hex": hx,
        "source": f"lexicon:{best.source}",
    }

    hue = _hex_to_hue_deg(hx)
    if hue is not None:
        info["hue_deg"] = hue

    lab = _hex_to_lab(hx)
    if lab is not None:
        _L, a, b = lab
        info["lab_hue_deg"] = _lab_hue_deg_from_ab(a, b)

    return info


# ---------------------------------------------------------------------------
# spaCy STOP_WORDS (lazy)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _stop_words_en() -> set[str]:
    spacy = require("spacy", extra="spacy", purpose="STOP_WORDS for style extraction.")
    return set(spacy.lang.en.stop_words.STOP_WORDS)


# ---------------------------------------------------------------------------
# Product filtering (style extraction)
# ---------------------------------------------------------------------------

PRODUCT_BASES: set[str] = {"lipstick"}


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


# ---------------------------------------------------------------------------
# Mention extraction (n-gram scan on normalized text)
# ---------------------------------------------------------------------------

def extract_mentions_free(doc_text: str, color_index: Dict[str, Dict[str, Any]]) -> List[Mention]:
    toks = _norm(doc_text).split()
    if not toks:
        return []

    max_n = _max_ngram_from_index(color_index)
    hits: List[Mention] = []
    i = 0

    while i < len(toks):
        matched: Optional[Mention] = None
        for n in range(min(max_n, len(toks) - i), 0, -1):
            cand = " ".join(toks[i : i + n])

            info = color_index.get(cand)
            if not info:
                info = _lexicon_info_for_candidate(cand)

            if info:
                matched = {
                    "alias": cand,
                    "name": str(info.get("name") or cand),
                    "tok_start": i,
                    "tok_len": n,
                }

                lab_h = _safe_float(info.get("lab_hue_deg"))
                if lab_h is not None:
                    matched["lab_hue_deg"] = lab_h
                else:
                    hue = _safe_float(info.get("hue_deg"))
                    if hue is not None:
                        matched["hue_deg"] = hue

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


# ---------------------------------------------------------------------------
# Style extraction (token frequency, stopwords filtered)
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
    return [
        StyleIntent(label=k, strength=v / mx, polarity="LIKE", source="GLOBAL")
        for k, v in counts.items()
    ]


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
    return [
        StyleIntent(label=k, strength=v / mx, polarity="AVOID", source="ELLIPTICAL_NEG")
        for k, v in counts.items()
    ]


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


# ---------------------------------------------------------------------------
# Elliptical segmentation helpers (kept, used by other modules historically)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def analyze_clauses_with_llm(
    chunks: List[ClauseChunk],
    color_index: Dict[str, Dict[str, Any]],
    *,
    llm_polarity_fn: PolarityLLM,
    infer_clause_sentiment_fn: Optional[Callable[[str], Optional[str]]] = None,
    expose_styles: bool = True,
    debug_mentions: bool = False,
) -> Dict[str, Any]:
    if llm_polarity_fn is None:
        raise RuntimeError("analyze_clauses_with_llm requires llm_polarity_fn.")

    likes: List[LikeItem] = []
    dislikes: List[DislikeItem] = []

    for ch in chunks:
        mentions = extract_mentions_free(ch["text"], color_index)
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
            # Keep backward compat: emit hue_deg if present (prefer lab_hue_deg).
            hue = m.get("lab_hue_deg", m.get("hue_deg"))
            if pols.get(nm) == "LIKE":
                it: LikeItem = {"family": nm}
                if hue is not None:
                    it["hue_deg"] = float(hue)
                likes.append(it)
            elif pols.get(nm) == "DISLIKE":
                it2: DislikeItem = {"family": nm}
                if hue is not None:
                    it2["hue_deg"] = float(hue)
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

    return out


__all__ = [
    "LikeItem",
    "DislikeItem",
    "StyleIntent",
    "Mention",
    "build_world_alias_index",
    "extract_mentions_free",
    "analyze_clauses_with_llm",
]
