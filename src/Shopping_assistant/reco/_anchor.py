# src/Shopping_assistant/reco/_anchor.py
from __future__ import annotations

import os

from Shopping_assistant.nlp.runtime.lexicon import load_default_lexicon

from ._colorconv import _hex_to_lab
from ._utils import _get, _get_enum_value
from ._constants import _LEX_TOPK, _LEX_FUZZY_CUTOFF, _LEX_SCORE_EPS


def _has_color_like_mention(nlp_res) -> bool:
    for m in _get(nlp_res, "mentions", ()) or ():
        kind = _get_enum_value(_get(m, "kind", None))
        if (kind or "").lower() != "colors":
            continue
        pol = _get_enum_value(_get(m, "polarity", None))
        if (pol or "").lower() in {"like", "neutral", "unknown"}:
            return True
    return False


def _seed_hex_from_nlp(nlp_res):
    best_hex = None
    best_conf = -1.0
    for m in _get(nlp_res, "mentions", ()) or ():
        kind = (_get_enum_value(_get(m, "kind", None)) or "").lower()
        if kind != "colors":
            continue
        pol = (_get_enum_value(_get(m, "polarity", None)) or "").lower()
        if pol not in {"like", "neutral", "unknown"}:
            continue
        meta = _get(m, "meta", {}) or {}
        if not isinstance(meta, dict):
            continue
        hx = meta.get("seed_hex") or meta.get("hex")
        if not (isinstance(hx, str) and hx.strip().startswith("#") and len(hx.strip()) == 7):
            continue
        conf = float(_get(m, "confidence", 0.0) or 0.0)
        if conf > best_conf:
            best_conf = conf
            best_hex = hx.strip()
    return best_hex


def _best_lexicon_lab_and_hex(lex, query: str):
    """
    Resolve top-k. Keep near-best score candidates (when score exists), then pick max chroma.
    Returns (lab, hex).
    """
    if lex is None:
        return None, None

    try:
        res = lex.resolve(
            query,
            topk=int(_LEX_TOPK),
            fuzzy_cutoff=float(_LEX_FUZZY_CUTOFF),
            use_semantic=True,
        )
    except Exception:
        res = []

    if not res:
        return None, None

    cands = []
    for r in res:
        hx = getattr(r, "hex", None)
        if not isinstance(hx, str) or not hx:
            continue
        lab = _hex_to_lab(hx)
        if lab is None:
            continue
        score = getattr(r, "score", None)
        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None
        L, a, b = lab
        C = float((a * a + b * b) ** 0.5)
        cands.append((score_f, C, lab, hx))

    if not cands:
        return None, None

    scores = [s for s, _, _, _ in cands if s is not None]
    if scores:
        best = max(scores)
        keep = [x for x in cands if x[0] is None or x[0] >= (best - float(_LEX_SCORE_EPS))]
    else:
        keep = cands

    keep.sort(key=lambda t: (t[1], -(t[0] if t[0] is not None else -1e9)), reverse=True)
    _, _, lab_best, hx_best = keep[0]
    return lab_best, hx_best


def _anchor_source_mode() -> str:
    """
    Controls anchor source to prevent polluted anchors.
    Values:
      - 'lexicon' (recommended for anchor tests; ignores meta seed/lab)
      - 'auto' (allows meta lab/seed then lexicon fallback)
    Default: 'lexicon'
    """
    v = (os.getenv("SA_ANCHOR_SOURCE", "lexicon") or "lexicon").strip().lower()
    return "auto" if v == "auto" else "lexicon"


def _anchor_from_nlp(nlp_res):
    """
    Returns (anchor_lab, anchor_hex_used)
    Priority depends on SA_ANCHOR_SOURCE:
      - lexicon: lexicon resolve(canonical/raw) only (ignores meta lab/seed_hex)
      - auto:
          1) meta lab_L/a/b
          2) meta seed_hex/hex
          3) lexicon resolve(canonical/raw)
    """
    best_lab = None
    best_hex = None
    best_score = -1.0

    try:
        lex = load_default_lexicon()
    except Exception:
        lex = None

    mode = _anchor_source_mode()

    for m in _get(nlp_res, "mentions", ()) or ():
        kind = _get_enum_value(_get(m, "kind", None))
        if (kind or "").lower() != "colors":
            continue

        pol = _get_enum_value(_get(m, "polarity", None))
        if (pol or "").lower() not in {"like", "neutral", "unknown"}:
            continue

        conf = float(_get(m, "confidence", 0.0) or 0.0)
        meta = _get(m, "meta", {}) or {}

        lab_m = None
        hex_m = None

        if mode == "auto":
            if isinstance(meta, dict):
                L0 = meta.get("lab_L")
                a0 = meta.get("lab_a")
                b0 = meta.get("lab_b")
                if isinstance(L0, (int, float)) and isinstance(a0, (int, float)) and isinstance(b0, (int, float)):
                    lab_m = (float(L0), float(a0), float(b0))

            if lab_m is None and isinstance(meta, dict):
                hx0 = meta.get("seed_hex") or meta.get("hex")
                if isinstance(hx0, str) and hx0:
                    hex_m = hx0
                    lab_m = _hex_to_lab(hx0)

        if lab_m is None and lex is not None:
            canon = str(_get(m, "canonical", "") or "").strip()
            raw = str(_get(m, "raw", "") or "").strip()
            query = (canon or raw).strip().lower()
            if query:
                lab_m, hex_m = _best_lexicon_lab_and_hex(lex, query)

        if lab_m is None:
            continue

        if conf > best_score:
            best_score = conf
            best_lab = lab_m
            best_hex = hex_m

    return best_lab, best_hex


def _first_color_token_from_nlp(nlp_res) -> str | None:
    """
    First color token from NLP mentions (canonical/raw).
    """
    for m in _get(nlp_res, "mentions", ()) or ():
        kind = (_get_enum_value(_get(m, "kind", None)) or "").lower()
        if kind != "colors":
            continue
        pol = (_get_enum_value(_get(m, "polarity", None)) or "").lower()
        if pol not in {"like", "neutral", "unknown"}:
            continue

        canon = str(_get(m, "canonical", "") or "").strip().lower()
        raw = str(_get(m, "raw", "") or "").strip().lower()
        return canon or raw or None
    return None


def _is_plain_color_query(nlp_res) -> bool:
    if not _has_color_like_mention(nlp_res):
        return False

    n = 0
    for m in _get(nlp_res, "mentions", ()) or ():
        kind = (_get_enum_value(_get(m, "kind", None)) or "").lower()
        if kind != "colors":
            continue
        pol = (_get_enum_value(_get(m, "polarity", None)) or "").lower()
        if pol in {"like", "neutral", "unknown"}:
            n += 1
    return n == 1
