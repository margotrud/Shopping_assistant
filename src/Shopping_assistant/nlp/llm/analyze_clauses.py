# src/Shopping_assistant/nlp/analyze_clauses_llm.py
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from Shopping_assistant.utils.optional_deps import require

from Shopping_assistant.nlp.parsing.clauses import ClauseChunk
from Shopping_assistant.nlp.parsing.polarity import PolarityLLM, infer_polarity_for_mentions

from typing import Any, Dict, Iterable, List, Tuple

# --- optional deps (already in your file): require(...)
# require = imported above from Shopping_assistant.utils.optional_deps

def _iter_css_color_names() -> Iterable[str]:
    webcolors = require("webcolors", extra="webcolors", purpose="CSS color name inventory")
    # webcolors API differs by version; support both.
    if hasattr(webcolors, "names"):
        # newer: webcolors.names("css3")
        try:
            return list(webcolors.names("css3"))
        except Exception:
            pass
    # older: webcolors.CSS3_NAMES_TO_HEX
    if hasattr(webcolors, "CSS3_NAMES_TO_HEX"):
        return list(getattr(webcolors, "CSS3_NAMES_TO_HEX").keys())
    return []


def _iter_xkcd_color_names() -> Iterable[str]:
    mpl = require("matplotlib", extra="matplotlib", purpose="XKCD color name inventory")
    from matplotlib import colors as mcolors  # type: ignore
    # keys look like "xkcd:cloudy blue" -> normalize later
    return list(getattr(mcolors, "XKCD_COLORS").keys())


def build_webcolor_world_palette(*, include_xkcd: bool = True) -> List[str]:
    """
    Does:
        Return a list of world color names (CSS + optional XKCD), dependency-driven.
    """
    names: List[str] = []
    names.extend(_iter_css_color_names())
    if include_xkcd:
        names.extend(_iter_xkcd_color_names())
    return names


def build_alias_index(names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """
    Does:
        Build alias_index: normalized alias -> info dict.
        No hardcoded vocab; caller provides names from deps/data.
    """
    def _norm(s: str) -> str:
        # normalize "xkcd:cloudy blue" -> "cloudy blue"
        s = s.lower().strip()
        if s.startswith("xkcd:"):
            s = s[5:]
        # keep alnum and spaces
        out = []
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

    idx: Dict[str, Dict[str, Any]] = {}
    for n in names:
        a = _norm(n)
        if not a:
            continue
        # minimal info; your extractor only needs "name" + alias key
        idx[a] = {"name": a, "source": "world"}
    return idx


def build_world_alias_index(*, include_xkcd: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Does:
        Build a dynamic alias index from CSS/XKCD color inventories.
    """
    return build_alias_index(build_webcolor_world_palette(include_xkcd=include_xkcd))


# ---------------------------------------------------------------------------
# spaCy STOP_WORDS (lazy)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _stop_words_en() -> set[str]:
    spacy = require(
        "spacy",
        extra="spacy",
        purpose="Needed for STOP_WORDS in analyze_clauses_llm.",
    )
    return set(spacy.lang.en.stop_words.STOP_WORDS)


# ---------------------------------------------------------------------------
# Config
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
# Typed outputs
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
    tok_start: int
    tok_len: int


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in s).split())


def _norm_tokens_with_spans(s: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    toks: List[str] = []
    spans: List[Tuple[int, int]] = []
    cur: List[str] = []
    start: Optional[int] = None

    for i, ch in enumerate(s):
        c = ch.lower() if ch.isalnum() else " "
        if c == " ":
            if cur:
                toks.append("".join(cur))
                spans.append((start if start is not None else i, i))
                cur = []
                start = None
            continue
        if start is None:
            start = i
        cur.append(c)

    if cur:
        toks.append("".join(cur))
        spans.append((start if start is not None else len(s), len(s)))

    return toks, spans


# ---------------------------------------------------------------------------
# World alias index
# ---------------------------------------------------------------------------

def build_world_alias_index(*, include_xkcd: bool = True) -> Dict[str, Dict[str, Any]]:
    webcolors = require("webcolors", extra="webcolors", purpose="CSS color inventory")
    idx: Dict[str, Dict[str, Any]] = {}

    def _norm(s: str) -> str:
        s = s.lower().strip()
        if s.startswith("xkcd:"):
            s = s[5:]
        out = []
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

    # CSS names -> hex
    css_names = []
    if hasattr(webcolors, "names"):
        try:
            css_names = list(webcolors.names("css3"))
        except Exception:
            css_names = []
    if not css_names and hasattr(webcolors, "CSS3_NAMES_TO_HEX"):
        css_names = list(getattr(webcolors, "CSS3_NAMES_TO_HEX").keys())

    for name in css_names:
        alias = _norm(name)
        if not alias:
            continue
        try:
            hx = webcolors.name_to_hex(name)
        except Exception:
            continue
        idx[alias] = {"name": alias, "hex": hx, "source": "css"}

    # XKCD names -> hex
    if include_xkcd:
        mpl = require("matplotlib", extra="matplotlib", purpose="XKCD color inventory")
        from matplotlib import colors as mcolors  # type: ignore
        for k, hx in getattr(mcolors, "XKCD_COLORS").items():
            alias = _norm(k)
            if not alias:
                continue
            idx[alias] = {"name": alias, "hex": hx, "source": "xkcd"}

    return idx

def _max_ngram_from_index(color_index: Dict[str, Dict[str, Any]]) -> int:
    return min(max(len(a.split()) for a in color_index.keys()), 4) if color_index else 1


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Mention extraction
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
            cand = " ".join(toks[i:i+n])
            info = color_index.get(cand)
            if info:
                matched = {
                    "alias": cand,
                    "name": str(info.get("name") or cand),
                    "tok_start": i,
                    "tok_len": n,
                }
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
# Style extraction (STOP_WORDS now lazy)
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
# Elliptical segmentation helpers (inchangé)
# ---------------------------------------------------------------------------

_NOT_START_RE = re.compile(
    r"(?:(?P<sep>[,;:\(\)\[\]\-–—])\s*|\bbut\s+)\bnot\b\s+",
    flags=re.IGNORECASE,
)


def _find_elliptical_split(text: str) -> Optional[int]:
    m = _NOT_START_RE.search(text)
    if not m:
        return None
    inner = re.search(r"\bnot\b", text[m.start():m.end()], flags=re.IGNORECASE)
    return None if not inner else m.start() + inner.start()


# ---------------------------------------------------------------------------
# Main API (inchangé)
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
            hue = m.get("hue_deg")
            if pols.get(nm) == "LIKE":
                it: LikeItem = {"family": nm}
                if hue is not None:
                    it["hue_deg"] = hue
                likes.append(it)
            elif pols.get(nm) == "DISLIKE":
                it2: DislikeItem = {"family": nm}
                if hue is not None:
                    it2["hue_deg"] = hue
                dislikes.append(it2)

    out: Dict[str, Any] = {
        "LIKES": likes,
        "DISLIKES": dislikes,
    }

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
