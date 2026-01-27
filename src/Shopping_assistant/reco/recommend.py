# src/Shopping_assistant/reco/recommend.py
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from Shopping_assistant.color.hard_color_pool import hard_color_pool, params_from_env
from Shopping_assistant.color.scoring import QuerySpec, score_shades
from Shopping_assistant.io.assets import AssetBundle, load_default_assets
from Shopping_assistant.nlp.interpretation.preference import interpret_nlp
from Shopping_assistant.nlp.runtime.lexicon import load_default_lexicon

# =============================================================================
# CONSTANTS
# =============================================================================

_DEFAULT_POOL_TOPN = 1200
_MAX_POOL_TOPN = 2000

# =============================================================================
# PRODUCT ELIGIBILITY FILTER
# =============================================================================

_BAD_PRODUCT_RE = re.compile(
    r"(?:\bclear\b|\buniversal\b|\brecharge\b|\brefill\b|\br?echargeable\b|\bécrin\b|\becrin\b|\bétui\b|\bcase\b|\bcap\b|\bmarbre\b|\bstrass\b)",
    re.I,
)

# =============================================================================
# HELPERS (generic, deterministic)
# =============================================================================


def _get(m, key: str, default=None):
    try:
        if isinstance(m, dict):
            return m.get(key, default)
        return getattr(m, key, default)
    except Exception:
        return default


def _get_enum_value(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            return x
        return getattr(x, "value", None)
    except Exception:
        return None


def _has_color_like_mention(nlp_res) -> bool:
    for m in _get(nlp_res, "mentions", ()) or ():
        kind = _get_enum_value(_get(m, "kind", None))
        if (kind or "").lower() != "color":
            continue
        pol = _get_enum_value(_get(m, "polarity", None))
        if (pol or "").lower() in {"like", "neutral", "unknown"}:
            return True
    return False


def _hex_to_rgb01(hx: str):
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


def _srgb01_to_linear(x: float) -> float:
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def _hex_to_lab(hx: str):
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

    fx, fy, fz = f(float(x)), f(float(y)), f(float(z))
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b2 = 200 * (fy - fz)
    return (float(L), float(a), float(b2))


def _seed_hex_from_nlp(nlp_res):
    best_hex = None
    best_conf = -1.0
    for m in _get(nlp_res, "mentions", ()) or ():
        kind = (_get_enum_value(_get(m, "kind", None)) or "").lower()
        if kind != "color":
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


def _anchor_lab_from_nlp(nlp_res):
    """
    Robust anchor (deterministic):
    1) meta lab_L/a/b if present
    2) mention seed_hex/hex
    3) lexicon resolve(canonical/raw) -> hex -> lab
    """
    best_lab = None
    best_score = -1.0

    try:
        lex = load_default_lexicon()
    except Exception:
        lex = None

    for m in _get(nlp_res, "mentions", ()) or ():
        kind = _get_enum_value(_get(m, "kind", None))
        if (kind or "").lower() != "color":
            continue

        pol = _get_enum_value(_get(m, "polarity", None))
        if (pol or "").lower() not in {"like", "neutral", "unknown"}:
            continue

        conf = float(_get(m, "confidence", 0.0) or 0.0)
        meta = _get(m, "meta", {}) or {}

        lab_m = None
        if isinstance(meta, dict):
            L0 = meta.get("lab_L")
            a0 = meta.get("lab_a")
            b0 = meta.get("lab_b")
            if isinstance(L0, (int, float)) and isinstance(a0, (int, float)) and isinstance(b0, (int, float)):
                lab_m = (float(L0), float(a0), float(b0))

        if lab_m is None and isinstance(meta, dict):
            hx0 = meta.get("seed_hex") or meta.get("hex")
            if isinstance(hx0, str) and hx0:
                lab_m = _hex_to_lab(hx0)

        if lab_m is None and lex is not None:
            canon = str(_get(m, "canonical", "") or "").strip()
            raw = str(_get(m, "raw", "") or "").strip()
            query = (canon or raw).strip().lower()
            if query:
                try:
                    res = lex.resolve(query, topk=1, fuzzy_cutoff=75.0, use_semantic=False)
                except Exception:
                    res = []
                if res:
                    hx = getattr(res[0], "hex", None)
                    if isinstance(hx, str) and hx:
                        lab_m = _hex_to_lab(hx)

        if lab_m is None:
            continue

        if conf > best_score:
            best_score = conf
            best_lab = lab_m

    return best_lab


def _filter_invalid_products(inv: pd.DataFrame) -> pd.DataFrame:
    if inv is None or inv.empty:
        return inv
    txt = (inv.get("product_name", "").astype(str) + " " + inv.get("shade_name", "").astype(str)).fillna("")
    keep = ~txt.str.contains(_BAD_PRODUCT_RE, regex=True)
    return inv.loc[keep].copy()


def _restore_lab_cols(df_pool: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """
    score_shades may drop Lab columns. Restore them deterministically.
    """
    if scored is None or scored.empty or df_pool is None or df_pool.empty:
        return scored

    want = [
        c
        for c in ("L_lab", "a_lab", "b_lab", "chip_hex", "_de00_anchor")
        if c in df_pool.columns and c not in scored.columns
    ]
    if not want:
        return scored

    key_candidates = [("product_id", "shade_id"), ("shade_id",), ("product_id",)]
    keys = None
    for kc in key_candidates:
        if all(k in df_pool.columns for k in kc) and all(k in scored.columns for k in kc):
            keys = list(kc)
            break

    if keys is None:
        if df_pool.index.equals(scored.index):
            out = scored.copy()
            for c in want:
                out[c] = df_pool[c]
            return out
        return scored

    left = scored.copy()
    right = df_pool[keys + want].drop_duplicates(subset=keys).copy()

    for k in keys:
        left[k] = left[k].astype(str)
        right[k] = right[k].astype(str)

    return left.merge(right, on=keys, how="left")


# =============================================================================
# MAIN
# =============================================================================


def recommend_from_text(
    text: str,
    *,
    assets: AssetBundle | None = None,
    topk: int = 20,
    candidate_pool_topn: int = _DEFAULT_POOL_TOPN,
    lambda_constraints: float = 2.0,
    lambda_preference: float = 0.0,
    debug: bool = False,
) -> pd.DataFrame:
    if assets is None:
        assets = load_default_assets()

    inv = assets.inventory.copy()
    inv = _filter_invalid_products(inv)

    pool_cap = int(min(max(int(candidate_pool_topn), 1), len(inv))) if len(inv) else 0
    pool_cap = int(min(pool_cap, int(_MAX_POOL_TOPN)))

    nlp_res = interpret_nlp(text, debug=debug)
    has_color = _has_color_like_mention(nlp_res)

    seed_hex_nlp = _seed_hex_from_nlp(nlp_res)
    anchor_lab = _anchor_lab_from_nlp(nlp_res)
    if anchor_lab is None and isinstance(seed_hex_nlp, str):
        anchor_lab = _hex_to_lab(seed_hex_nlp)

    # ==========================
    # PHASE A — strict pool
    # ==========================
    if has_color and anchor_lab is not None:
        p = params_from_env()
        df_pool = hard_color_pool(inv, anchor_lab=anchor_lab, params=p)

        # compute cap (does not break purity)
        if pool_cap > 0 and len(df_pool) > pool_cap:
            df_pool = df_pool.head(pool_cap).copy()

        if debug:
            aC = float(np.hypot(float(anchor_lab[1]), float(anchor_lab[2])))
            thr_used = float(p.de00_max_neutral if aC < float(p.neutral_anchor_c_max) else p.de00_max)
            print("\n[PHASE A]")
            print(f"  text={text!r}")
            print(f"  has_color={has_color}  anchor_lab={anchor_lab}  anchor_C={aC:.2f}")
            print(f"  de00_thr={thr_used:.2f}  pool_cap={pool_cap}  pool_n={len(df_pool)}  inv_n={len(inv)}")

        # strict contract: can be empty, and that's OK
        if df_pool.empty:
            return df_pool
    else:
        # not a color query: just compute-cap (not Phase A)
        df_pool = inv.head(pool_cap).copy()
        if debug:
            print("\n[PHASE A]")
            print(f"  text={text!r}")
            print(f"  has_color={has_color}  anchor_lab={anchor_lab}")
            print(f"  pool_cap={pool_cap}  pool_n={len(df_pool)}  inv_n={len(inv)}")

    # ==========================
    # Phase B scoring (unchanged)
    # ==========================
    query = QuerySpec(anchor_lab=anchor_lab)
    scored = score_shades(
        df_pool,
        None,
        query,
        lambda_constraints=float(lambda_constraints),
        lambda_preference=float(lambda_preference),
        calibration=assets.calibration,
        preference_weights=getattr(assets, "preference_weights", None),
    )

    scored = _restore_lab_cols(df_pool, scored)

    score_col = "score_total" if "score_total" in scored.columns else ("score" if "score" in scored.columns else None)
    if score_col is None:
        return scored.head(int(topk)).copy()

    return scored.sort_values(score_col, ascending=False).head(int(topk)).copy()


__all__ = ["recommend_from_text"]
