# src/Shopping_assistant/reco/recommend.py
from __future__ import annotations

import colorsys
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from Shopping_assistant.color.deltae import delta_e_ciede2000
from Shopping_assistant.color.hard_color_pool import hard_color_pool, params_from_env
from Shopping_assistant.color.scoring import QuerySpec, score_shades
from Shopping_assistant.io.assets import AssetBundle, load_default_assets
from Shopping_assistant.nlp.interpretation.preference import interpret_nlp
from Shopping_assistant.nlp.runtime.lexicon import load_default_lexicon

# within-family constraints (label-relative)
from Shopping_assistant.color.constraints import (
    ConstraintSpec as FamilyConstraintSpec,
    load_label_distributions,
)

# =============================================================================
# CONSTANTS
# =============================================================================

_DEFAULT_POOL_TOPN = 1200
_MAX_POOL_TOPN = 2000

# Hue-family fallback (generic; restrict pool for low-chroma anchors; MUST NOT rewrite anchor)
_HUE_FALLBACK_BAND_DEG = 28.0
_HUE_FALLBACK_MIN_POOL_N = 120  # used as "support is weak" threshold (coverage gate)
_HUE_FALLBACK_CHROMA_Q = 0.50  # kept for debug/telemetry; not used to rewrite anchor
_HUE_FALLBACK_ANCHOR_C_MIN = 55.0

# Lexicon anchor selection (generic)
_LEX_TOPK = 10
_LEX_FUZZY_CUTOFF = 60.0
_LEX_SCORE_EPS = 0.03  # keep near-best score candidates when score is available

# within-family constraint gating
_FAMILY_P_MIN = 0.55
_FAMILY_P_HI = 0.75
_FAMILY_FLOOR = 0.35

# quantiles supported by label_distributions (prevents KeyError like 0.65 missing)
_ALLOWED_Q = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=float)

# Adaptive pool fallback when strict ΔE yields empty pool
_POOL_FALLBACK_MIN_N = 10
_POOL_FALLBACK_DE00_STEPS = (0.0, 2.0, 4.0, 8.0, 12.0)  # added to thr_used
_POOL_FALLBACK_DE00_CAP = 30.0

# High-chroma hue-band fallback when ΔE cannot find any support
_HUE_FAILSAFE_BAND_DEG = 32.0

# Naming-prob pool fallback (when anchor has weak dataset support at strict ΔE)
# No hardcoding: we use p_<label> if it exists in the parquet.
_NAMING_POOL_MIN_N = 20
_NAMING_POOL_TOPN = 180
_NAMING_POOL_P_MIN = 0.20

# Domain anchor (dataset-driven) using naming probs p_<label>
_DOMAIN_ANCHOR_P_MIN = 0.55
_DOMAIN_ANCHOR_MIN_N = 40
_DOMAIN_ANCHOR_TOPN = 250

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
        if (kind or "").lower() != "colors":
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


def _hex_to_hsv_h_deg(hx: str):
    rgb01 = _hex_to_rgb01(hx)
    if rgb01 is None:
        return None
    r, g, b = rgb01
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return float(h * 360.0)


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
        C = float(np.hypot(a, b))
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


def _filter_invalid_products(inv: pd.DataFrame) -> pd.DataFrame:
    if inv is None or inv.empty:
        return inv
    txt = (inv.get("product_name", "").astype(str) + " " + inv.get("shade_name", "").astype(str)).fillna("")
    keep = ~txt.str.contains(_BAD_PRODUCT_RE, regex=True)
    return inv.loc[keep].copy()


def _restore_lab_cols(df_pool: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
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


def _circ_dist_deg(a_deg: np.ndarray, b0_deg: float) -> np.ndarray:
    d = (a_deg - float(b0_deg) + 180.0) % 360.0 - 180.0
    return np.abs(d)


def _ensure_hue_rgb_deg(inv: pd.DataFrame) -> pd.DataFrame:
    """
    Adds column _H_rgb_deg computed from chip_hex (HSV hue in degrees) if not present.
    Safe for small inventories (<= few 10k). Your inv is ~1.6k.
    """
    if inv is None or inv.empty:
        return inv
    if "_H_rgb_deg" in inv.columns:
        return inv
    out = inv.copy()
    if "chip_hex" not in out.columns:
        return out
    out["_H_rgb_deg"] = out["chip_hex"].astype(str).map(_hex_to_hsv_h_deg)
    return out


def _domain_pool_and_anchor(
    inv: pd.DataFrame,
    *,
    anchor_lab: tuple[float, float, float],
    anchor_hex: str | None,
    hue_band_deg: float = _HUE_FALLBACK_BAND_DEG,
    min_pool_n: int = _HUE_FALLBACK_MIN_POOL_N,
    chroma_q: float = _HUE_FALLBACK_CHROMA_Q,
    anchor_c_min: float = _HUE_FALLBACK_ANCHOR_C_MIN,
) -> tuple[pd.DataFrame, tuple[float, float, float], dict]:
    """
    Hue-family fallback when anchor is low-chroma.

    IMPORTANT:
      - Hue is computed in HSV/RGB space (stable for light pinks).
      - Fallback MUST ONLY restrict the candidate pool.
      - Fallback MUST NOT rewrite/boost the anchor (nude/beige are valid low-chroma targets).
    """
    dbg = {
        "fallback_used": False,
        "anchor_C": None,
        "H0": None,
        "cand_n": None,
        "cthr": None,  # kept for backward debug printouts; always None now
        "anchor_lab_eff": None,
        "hue_source": None,
    }

    if inv is None or inv.empty:
        return inv, anchor_lab, dbg

    if not {"L_lab", "a_lab", "b_lab", "C_lab"}.issubset(inv.columns):
        return inv, anchor_lab, dbg

    L0, a0, b0 = map(float, anchor_lab)
    C0 = float(np.hypot(a0, b0))
    dbg["anchor_C"] = C0

    if not (C0 < float(anchor_c_min)):
        return inv, anchor_lab, dbg

    inv2 = _ensure_hue_rgb_deg(inv)
    H0 = _hex_to_hsv_h_deg(anchor_hex) if isinstance(anchor_hex, str) else None
    if H0 is None:
        # last resort: if no hex, fall back to Lab-hue (can be wrong for light pinks)
        H0 = float(np.degrees(np.arctan2(b0, a0)) % 360.0)
        dbg["hue_source"] = "lab"
    else:
        dbg["hue_source"] = "rgb"

    dbg["H0"] = float(H0)

    if "_H_rgb_deg" in inv2.columns and pd.to_numeric(inv2["_H_rgb_deg"], errors="coerce").notna().any():
        hvals = pd.to_numeric(inv2["_H_rgb_deg"], errors="coerce").to_numpy(float)
        hdist = _circ_dist_deg(hvals, float(H0))
        hue_mask = np.isfinite(hdist) & (hdist <= float(hue_band_deg))
        cand = inv2.loc[hue_mask].copy()
    else:
        if "H_lab_deg" not in inv2.columns:
            return inv, anchor_lab, dbg
        hdist = _circ_dist_deg(inv2["H_lab_deg"].to_numpy(float), float(H0))
        cand = inv2.loc[hdist <= float(hue_band_deg)].copy()

    dbg["cand_n"] = int(len(cand))
    if cand.empty:
        return inv, anchor_lab, dbg

    dbg["fallback_used"] = True
    dbg["anchor_lab_eff"] = tuple(map(float, anchor_lab))
    return cand, tuple(map(float, anchor_lab)), dbg


def _de00_to_anchor(inv: pd.DataFrame, anchor_lab: tuple[float, float, float]) -> np.ndarray:
    L0, a0, b0 = map(float, anchor_lab)
    L = inv["L_lab"].to_numpy(float)
    a = inv["a_lab"].to_numpy(float)
    b = inv["b_lab"].to_numpy(float)
    m = np.isfinite(L) & np.isfinite(a) & np.isfinite(b)
    d = np.full(len(inv), np.inf, float)
    if np.any(m):
        d[m] = delta_e_ciede2000(L[m], a[m], b[m], L0, a0, b0)
    return d


def _pool_by_de00_threshold(inv: pd.DataFrame, d: np.ndarray, thr: float) -> pd.DataFrame:
    m = np.isfinite(d) & (d <= float(thr))
    if not np.any(m):
        return inv.iloc[0:0].copy()
    out = inv.loc[m].copy()
    out["_de00_anchor"] = d[m]
    out = out.sort_values("_de00_anchor", ascending=True)
    return out


def _adaptive_de00_pool(
    inv: pd.DataFrame,
    *,
    anchor_lab: tuple[float, float, float],
    thr_base: float,
    min_n: int,
    cap: float,
) -> tuple[pd.DataFrame, dict]:
    dbg = {"adaptive_used": False, "thr_base": float(thr_base), "thr_final": None, "pool_n": 0}
    if inv is None or inv.empty:
        return inv, dbg

    d = _de00_to_anchor(inv, anchor_lab)

    best = inv.iloc[0:0].copy()
    for step in _POOL_FALLBACK_DE00_STEPS:
        thr = float(min(cap, float(thr_base) + float(step)))
        cand = _pool_by_de00_threshold(inv, d, thr)
        if len(cand) > 0:
            best = cand
            dbg["thr_final"] = thr
            dbg["pool_n"] = int(len(cand))
        if len(cand) >= int(min_n):
            dbg["adaptive_used"] = (step != 0.0)
            return cand, dbg

    if len(best) > 0:
        dbg["adaptive_used"] = True
        return best, dbg

    if np.isfinite(d).any():
        idx = np.argsort(d)
        idx = idx[np.isfinite(d[idx])]
        idx = idx[: max(int(min_n), 1)]
        out = inv.iloc[idx].copy()
        out["_de00_anchor"] = d[idx]
        out = out.sort_values("_de00_anchor", ascending=True)
        dbg["adaptive_used"] = True
        dbg["thr_final"] = None
        dbg["pool_n"] = int(len(out))
        return out, dbg

    return inv.iloc[0:0].copy(), dbg


def _hue_band_subset(
    inv: pd.DataFrame, *, anchor_hex: str | None, anchor_lab: tuple[float, float, float], band_deg: float
):
    inv2 = _ensure_hue_rgb_deg(inv)
    H0 = _hex_to_hsv_h_deg(anchor_hex) if isinstance(anchor_hex, str) else None
    if H0 is None:
        _, a0, b0 = map(float, anchor_lab)
        H0 = float(np.degrees(np.arctan2(b0, a0)) % 360.0)

    if "_H_rgb_deg" in inv2.columns and pd.to_numeric(inv2["_H_rgb_deg"], errors="coerce").notna().any():
        hvals = pd.to_numeric(inv2["_H_rgb_deg"], errors="coerce").to_numpy(float)
        hdist = _circ_dist_deg(hvals, float(H0))
        m = np.isfinite(hdist) & (hdist <= float(band_deg))
        return inv2.loc[m].copy(), {"H0": float(H0), "band": float(band_deg), "cand_n": int(np.sum(m))}
    return inv2.iloc[0:0].copy(), {"H0": float(H0), "band": float(band_deg), "cand_n": 0}


# =============================================================================
# naming probs + label distributions (cached)
# =============================================================================


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _load_chips_naming_probs() -> pd.DataFrame:
    root = _project_root()
    p = root / "data" / "colors" / "chips_with_naming_probs.parquet"
    if not p.exists():
        return pd.DataFrame()

    try:
        import pyarrow.parquet as pq

        table = pq.read_table(p)
        df = table.to_pandas()
    except Exception:
        df = pd.read_parquet(p)

    if "chip_hex" not in df.columns:
        return pd.DataFrame()

    pcols = [c for c in df.columns if c.startswith("p_")]
    keep = ["chip_hex"] + pcols
    out = df[keep].copy()
    out["chip_hex"] = out["chip_hex"].astype(str).str.lower()
    return out.drop_duplicates(subset=["chip_hex"])


@lru_cache(maxsize=1)
def _load_family_label_distributions() -> dict:
    root = _project_root()
    p = root / "data" / "colors" / "label_distributions.json"
    if not p.exists():
        return {}
    return load_label_distributions(p)


def _attach_naming_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merge p_<label> onto df using chip_hex.
    No hardcoding: uses whatever p_* exists in the parquet.
    """
    if df is None or df.empty:
        return df
    if "chip_hex" not in df.columns:
        return df

    probs = _load_chips_naming_probs()
    if probs.empty:
        return df

    out = df.copy()
    out["chip_hex"] = out["chip_hex"].astype(str).str.lower()

    if any(c.startswith("p_") for c in out.columns):
        return out

    return out.merge(probs, on="chip_hex", how="left")


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


def _naming_prob_label_supported(label: str, *, inv_with_probs: pd.DataFrame | None = None) -> bool:
    """
    Generic support check for label->p_<label> existence.
    No aliasing/hardcoding.
    """
    if not isinstance(label, str) or not label:
        return False
    pcol = f"p_{label.strip().lower()}"

    if inv_with_probs is not None and isinstance(inv_with_probs, pd.DataFrame) and (pcol in inv_with_probs.columns):
        return True

    probs = _load_chips_naming_probs()
    if probs.empty:
        return False
    return pcol in probs.columns


def _pool_by_naming_prob(
    inv: pd.DataFrame,
    *,
    label: str,
    pool_topn: int = _NAMING_POOL_TOPN,
    p_min: float = _NAMING_POOL_P_MIN,
    min_n: int = _NAMING_POOL_MIN_N,
) -> tuple[pd.DataFrame, dict]:
    """
    Build candidate pool by p_<label> if present.
    No aliasing/hardcode: label must match a parquet p_* column.
    Returns (pool, debug_dict).
    """
    dbg = {"used": False, "label": label, "pcol": None, "pool_n": 0, "p_min": float(p_min), "topn": int(pool_topn)}
    if inv is None or inv.empty:
        return inv, dbg
    if not isinstance(label, str) or not label:
        return inv.iloc[0:0].copy(), dbg

    pcol = f"p_{label.strip().lower()}"
    dbg["pcol"] = pcol

    inv2 = _attach_naming_probs(inv)
    if pcol not in inv2.columns:
        return inv2.iloc[0:0].copy(), dbg

    tmp = inv2.copy()
    tmp[pcol] = pd.to_numeric(tmp[pcol], errors="coerce").fillna(0.0)
    tmp = tmp.sort_values(pcol, ascending=False)

    cand = tmp.loc[tmp[pcol] >= float(p_min)].copy()
    if len(cand) < int(min_n):
        cand = tmp.head(int(pool_topn)).copy()

    dbg["used"] = True
    dbg["pool_n"] = int(len(cand))
    return cand, dbg


def _first_family_label_from_nlp(
    nlp_res,
    *,
    dists: dict,
    inv_with_probs: pd.DataFrame | None = None,
) -> tuple[str | None, bool]:
    """
    Returns (family_label, label_in_distributions).

    Priority:
      1) if token matches label_distributions key -> use it (label_in_distributions=True)
      2) else if token has p_<token> column available -> use it (label_in_distributions=False)
      3) else None
    No aliasing/hardcode.
    """
    tok = _first_color_token_from_nlp(nlp_res)
    if not tok:
        return None, False

    tok = tok.strip().lower()
    if dists:
        keys = set(str(k).lower() for k in dists.keys())
        if tok in keys:
            return tok, True

    if _naming_prob_label_supported(tok, inv_with_probs=inv_with_probs):
        return tok, False

    return None, False


def _is_plain_color_query(nlp_res) -> bool:
    cons = tuple(_get(nlp_res, "constraints", ()) or ())
    return _has_color_like_mention(nlp_res) and (len(cons) == 0)


def _domain_anchor_from_naming_probs(inv: pd.DataFrame, *, label: str, p_min: float) -> tuple[float, float, float] | None:
    """
    Generic domain anchor:
      - uses p_<label> from chips naming probs
      - selects either >= p_min or topN by probability
      - returns weighted centroid in Lab
    """
    if inv is None or inv.empty:
        return None
    if not {"L_lab", "a_lab", "b_lab", "chip_hex"}.issubset(inv.columns):
        return None

    pcol = f"p_{str(label).lower()}"
    if pcol not in inv.columns:
        return None

    w = pd.to_numeric(inv[pcol], errors="coerce").fillna(0.0).to_numpy(float)

    m = w >= float(p_min)
    if int(np.sum(m)) < int(_DOMAIN_ANCHOR_MIN_N):
        idx = np.argsort(-w)
        idx = idx[w[idx] > 0.0][: int(_DOMAIN_ANCHOR_TOPN)]
        if len(idx) < int(_DOMAIN_ANCHOR_MIN_N):
            return None
        sub = inv.iloc[idx].copy()
        wsub = w[idx]
    else:
        sub = inv.loc[m].copy()
        wsub = w[m]

    L = pd.to_numeric(sub["L_lab"], errors="coerce").to_numpy(float)
    a = pd.to_numeric(sub["a_lab"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(sub["b_lab"], errors="coerce").to_numpy(float)
    ok = np.isfinite(L) & np.isfinite(a) & np.isfinite(b) & np.isfinite(wsub) & (wsub > 0.0)
    if not np.any(ok):
        return None

    ww = wsub[ok]
    ww = ww / (np.sum(ww) + 1e-12)

    L0 = float(np.sum(L[ok] * ww))
    a0 = float(np.sum(a[ok] * ww))
    b0 = float(np.sum(b[ok] * ww))
    return (L0, a0, b0)


def _snap_q(q: float) -> float:
    qf = float(q)
    i = int(np.argmin(np.abs(_ALLOWED_Q - qf)))
    return float(_ALLOWED_Q[i])


def _family_specs_from_nlp(nlp_constraints: Sequence[object]) -> list[FamilyConstraintSpec]:
    """
    Map existing NLP constraint axes into within-family axes.
    Generic mapping:
      - brightness lower/raise -> L below/above
      - saturation/vibrancy lower/raise -> C below/above

    IMPORTANT: quantiles must exist in label_distributions; we snap to _ALLOWED_Q.
    """
    if not nlp_constraints:
        return []

    specs: list[FamilyConstraintSpec] = []

    def _strength_params(strength: str, direction: str):
        s = (strength or "med").lower()
        d = (direction or "").lower()

        s_w = {"weak": 0.6, "med": 0.85, "strong": 1.0}.get(s, 0.85)

        if d in {"raise", "higher", "up", "increase"}:
            if s == "weak":
                q_lo, q_hi = 0.45, 0.65
            elif s == "strong":
                q_lo, q_hi = 0.60, 0.85
            else:
                q_lo, q_hi = 0.50, 0.75
        else:
            if s == "weak":
                q_lo, q_hi = 0.35, 0.55
            elif s == "strong":
                q_lo, q_hi = 0.15, 0.40
            else:
                q_lo, q_hi = 0.25, 0.50

        q_lo_s = _snap_q(q_lo)
        q_hi_s = _snap_q(q_hi)
        if q_hi_s <= q_lo_s:
            idx = int(np.where(_ALLOWED_Q == q_lo_s)[0][0])
            q_hi_s = float(_ALLOWED_Q[min(idx + 1, len(_ALLOWED_Q) - 1)])
        return float(q_lo_s), float(q_hi_s), float(s_w)

    for c in nlp_constraints:
        axis = (_get_enum_value(_get(c, "axis", None)) or "").lower()
        direction = (_get_enum_value(_get(c, "direction", None)) or "").lower()
        strength = (_get_enum_value(_get(c, "strength", None)) or "med").lower()

        if axis not in {"brightness", "saturation", "vibrancy"}:
            continue
        if direction not in {"lower", "raise"}:
            continue

        q_lo, q_hi, s_w = _strength_params(strength, direction)

        if axis == "brightness":
            specs.append(
                FamilyConstraintSpec(
                    axis="L",
                    direction=("below" if direction == "lower" else "above"),
                    strength=float(s_w),
                    q_lo=float(q_lo),
                    q_hi=float(q_hi),
                )
            )
        else:
            specs.append(
                FamilyConstraintSpec(
                    axis="C",
                    direction=("below" if direction == "lower" else "above"),
                    strength=float(s_w),
                    q_lo=float(q_lo),
                    q_hi=float(q_hi),
                )
            )

    return specs


def _anchor_inventory_coverage(inv: pd.DataFrame, anchor_lab: tuple[float, float, float], thr: float):
    """
    Coverage gate (diagnostic + guardrail):
      - min_deltaE00 vs inventory
      - count <= thr

    This is NOT "inventory is truth"; it is "can this dataset support this anchor".
    """
    if inv is None or inv.empty:
        return np.inf, 0
    if not {"L_lab", "a_lab", "b_lab"}.issubset(inv.columns):
        return np.inf, 0

    L0, a0, b0 = map(float, anchor_lab)
    L = inv["L_lab"].to_numpy(float)
    a = inv["a_lab"].to_numpy(float)
    b = inv["b_lab"].to_numpy(float)
    m = np.isfinite(L) & np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.inf, 0

    d = np.full(len(inv), np.inf, float)
    d[m] = delta_e_ciede2000(L[m], a[m], b[m], L0, a0, b0)

    mn = float(np.min(d))
    n = int(np.sum(d <= float(thr)))
    return mn, n


# =============================================================================
# ANCHOR-ONLY API (no scoring, no inventory)
# =============================================================================


def resolve_anchor_from_text(text: str, *, debug: bool = False) -> dict:
    """
    Anchor-only resolution for "plain colors" debugging.
    Returns a dict with anchor_hex, anchor_lab, and minimal NLP context.

    NOTE:
      - Uses SA_ANCHOR_SOURCE ('lexicon' default) to prevent polluted anchors.
      - Does NOT score / does NOT touch inventory.
    """
    nlp_res = interpret_nlp(text, debug=debug)
    has_color = _has_color_like_mention(nlp_res)

    seed_hex_nlp = _seed_hex_from_nlp(nlp_res)
    anchor_lab, anchor_hex = _anchor_from_nlp(nlp_res)

    if anchor_hex is None:
        anchor_hex = seed_hex_nlp
    if anchor_lab is None and isinstance(seed_hex_nlp, str):
        anchor_lab = _hex_to_lab(seed_hex_nlp)

    out = {
        "text": text,
        "has_color": bool(has_color),
        "anchor_source": _anchor_source_mode(),
        "anchor_hex": anchor_hex,
        "anchor_lab": anchor_lab,
    }

    if debug:
        print("\n[ANCHOR-ONLY]")
        print(f"  text={text!r}")
        print(f"  has_color={has_color}")
        print(f"  anchor_source={out['anchor_source']!r}")
        print(f"  anchor_hex={anchor_hex}  anchor_lab={anchor_lab}")

    return out


def resolve_effective_anchor_from_text(
    text: str,
    *,
    assets: AssetBundle | None = None,
    debug: bool = False,
) -> dict:
    """
    Resolve BOTH:
      - lexicon anchor (same as resolve_anchor_from_text)
      - effective anchor used by recommend_from_text for plain-color queries
        (domain anchor via naming probs p_<label> when available)

    Returns:
      {
        anchor_hex,
        anchor_lab_lexicon,
        anchor_lab_effective,
        has_color,
        family_label,                 # naming-prob label if available (even if not in dists)
        family_label_in_dists (bool), # only True when label_distributions supports it
        used_domain_anchor (bool),
      }
    """
    if assets is None:
        assets = load_default_assets()

    nlp_res = interpret_nlp(text, debug=debug)
    has_color = _has_color_like_mention(nlp_res)

    seed_hex_nlp = _seed_hex_from_nlp(nlp_res)
    anchor_lab, anchor_hex = _anchor_from_nlp(nlp_res)
    if anchor_hex is None:
        anchor_hex = seed_hex_nlp
    if anchor_lab is None and isinstance(seed_hex_nlp, str):
        anchor_lab = _hex_to_lab(seed_hex_nlp)

    inv = assets.inventory.copy()
    inv = _filter_invalid_products(inv)
    inv = _attach_naming_probs(inv)

    dists = _load_family_label_distributions()
    family_label, in_dists = _first_family_label_from_nlp(nlp_res, dists=dists, inv_with_probs=inv) if has_color else (None, False)

    anchor_eff = anchor_lab
    used_domain = False

    if has_color and anchor_eff is not None and family_label is not None and _is_plain_color_query(nlp_res):
        dom = _domain_anchor_from_naming_probs(inv, label=family_label, p_min=float(_DOMAIN_ANCHOR_P_MIN))
        if dom is not None:
            anchor_eff = dom
            used_domain = True

    out = {
        "text": text,
        "has_color": bool(has_color),
        "anchor_source": _anchor_source_mode(),
        "anchor_hex": anchor_hex,
        "anchor_lab_lexicon": anchor_lab,
        "anchor_lab_effective": anchor_eff,
        "family_label": family_label,
        "family_label_in_dists": bool(in_dists),
        "used_domain_anchor": bool(used_domain),
    }

    if debug:
        print("\n[ANCHOR-EFFECTIVE]")
        print(f"  text={text!r}")
        print(f"  has_color={has_color}")
        print(f"  family_label={family_label!r}  in_dists={in_dists}")
        print(f"  anchor_hex={anchor_hex}")
        print(f"  anchor_lab_lexicon={anchor_lab}")
        print(f"  anchor_lab_effective={anchor_eff}")
        print(f"  used_domain_anchor={used_domain}")

    return out


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
    """
    Recommendation mode (scoring).
    If you want only anchors (no pool, no scoring), call resolve_anchor_from_text().
    """
    if assets is None:
        assets = load_default_assets()

    inv = assets.inventory.copy()
    inv = _filter_invalid_products(inv)

    pool_cap = int(min(max(int(candidate_pool_topn), 1), len(inv))) if len(inv) else 0
    pool_cap = int(min(pool_cap, int(_MAX_POOL_TOPN)))

    nlp_res = interpret_nlp(text, debug=debug)
    has_color = _has_color_like_mention(nlp_res)

    seed_hex_nlp = _seed_hex_from_nlp(nlp_res)
    anchor_lab, anchor_hex = _anchor_from_nlp(nlp_res)
    if anchor_hex is None:
        anchor_hex = seed_hex_nlp
    if anchor_lab is None and isinstance(seed_hex_nlp, str):
        anchor_lab = _hex_to_lab(seed_hex_nlp)

    anchor_eff = anchor_lab

    # Attach naming probs onto inventory once (needed for domain anchor + naming pool + family constraints)
    inv = _attach_naming_probs(inv)

    # Precompute label_distributions + family label once (reuse later; avoid inconsistencies)
    dists = _load_family_label_distributions()
    family_label, family_in_dists = _first_family_label_from_nlp(nlp_res, dists=dists, inv_with_probs=inv) if has_color else (None, False)

    # Domain anchor for plain color queries: replace anchor_eff (pool+scoring) with dataset-driven centroid
    if has_color and anchor_eff is not None and family_label is not None and _is_plain_color_query(nlp_res):
        dom = _domain_anchor_from_naming_probs(inv, label=family_label, p_min=float(_DOMAIN_ANCHOR_P_MIN))
        if dom is not None:
            anchor_eff = dom
            if debug:
                print("\n[DOMAIN ANCHOR]")
                print(f"  family_label={family_label!r} p_min={_DOMAIN_ANCHOR_P_MIN} in_dists={family_in_dists}")
                print(f"  anchor_lab_lexicon={anchor_lab}")
                print(f"  anchor_lab_domain ={anchor_eff}")

    # ==========================
    # PHASE A — strict pool (+ hue-family fallback) + failsafes
    # ==========================
    if has_color and anchor_eff is not None:
        p = params_from_env()

        aC_raw = float(np.hypot(float(anchor_eff[1]), float(anchor_eff[2])))
        thr_used = float(p.de00_max_neutral if aC_raw < float(p.neutral_anchor_c_max) else p.de00_max)

        # coverage gate BEFORE any restriction
        min_de_all, n_support_all = _anchor_inventory_coverage(inv, tuple(map(float, anchor_eff)), thr_used)

        # only restrict by hue when dataset support is weak at the strict threshold
        if n_support_all < int(_HUE_FALLBACK_MIN_POOL_N):
            inv_eff, anchor_eff2, fb_dbg = _domain_pool_and_anchor(
                inv,
                anchor_lab=tuple(map(float, anchor_eff)),
                anchor_hex=anchor_hex,
            )
        else:
            inv_eff, anchor_eff2, fb_dbg = inv, tuple(map(float, anchor_eff)), {
                "fallback_used": False,
                "anchor_C": float(np.hypot(float(anchor_eff[1]), float(anchor_eff[2]))),
                "H0": None,
                "cand_n": None,
                "cthr": None,
                "anchor_lab_eff": tuple(map(float, anchor_eff)),
                "hue_source": None,
            }

        anchor_eff = anchor_eff2

        # coverage gate AFTER restriction (this is what truly matters for pool)
        min_de_eff, n_support_eff = _anchor_inventory_coverage(inv_eff, tuple(map(float, anchor_eff)), thr_used)

        df_pool = hard_color_pool(inv_eff, anchor_lab=anchor_eff, params=p)

        if pool_cap > 0 and len(df_pool) > pool_cap:
            df_pool = df_pool.head(pool_cap).copy()

        if debug:
            aC = float(np.hypot(float(anchor_eff[1]), float(anchor_eff[2])))
            print("\n[PHASE A]")
            print(f"  text={text!r}")
            print(f"  has_color={has_color}")
            print(
                f"  anchor_source={_anchor_source_mode()!r}  "
                f"anchor_lab_raw={anchor_lab}  anchor_hex={anchor_hex}  "
                f"anchor_lab_eff={anchor_eff}  anchor_C_eff={aC:.2f}"
            )
            print(
                f"  hue_fallback_used={bool(fb_dbg.get('fallback_used'))}"
                f"  hue_source={fb_dbg.get('hue_source')}"
                f"  hue_band={_HUE_FALLBACK_BAND_DEG}  chroma_q={_HUE_FALLBACK_CHROMA_Q}"
            )
            if fb_dbg.get("fallback_used"):
                h0 = fb_dbg.get("H0")
                if h0 is not None:
                    print(f"  hue_H0={float(h0):.2f}  cand_n={fb_dbg.get('cand_n')}  cthr={fb_dbg.get('cthr')}")
                else:
                    print(f"  hue_H0=None  cand_n={fb_dbg.get('cand_n')}  cthr={fb_dbg.get('cthr')}")
            print(
                f"  de00_thr={thr_used:.2f}  pool_cap={pool_cap}  pool_n={len(df_pool)}  "
                f"inv_n={len(inv)}  inv_eff_n={len(inv_eff)}"
            )
            print(f"  coverage_all: min_de00={min_de_all:.2f}  n_support<=thr={n_support_all}")
            print(f"  coverage_eff: min_de00={min_de_eff:.2f}  n_support<=thr={n_support_eff}")

        # FAILSAFE PATHS
        if df_pool.empty or (len(df_pool) < int(_POOL_FALLBACK_MIN_N)):
            # Fallback 1: adaptive ΔE pool on inv_eff (generic, no anchor rewrite)
            df_pool2, adbg = _adaptive_de00_pool(
                inv_eff,
                anchor_lab=tuple(map(float, anchor_eff)),
                thr_base=float(thr_used),
                min_n=int(_POOL_FALLBACK_MIN_N),
                cap=float(_POOL_FALLBACK_DE00_CAP),
            )

            # Fallback 2: if still empty/tiny -> hue-band subset then adaptive ΔE
            if df_pool2.empty or len(df_pool2) < int(_POOL_FALLBACK_MIN_N):
                hue_cand, hdbg = _hue_band_subset(
                    inv_eff,
                    anchor_hex=anchor_hex,
                    anchor_lab=tuple(map(float, anchor_eff)),
                    band_deg=float(_HUE_FAILSAFE_BAND_DEG),
                )
                if len(hue_cand) > 0:
                    df_pool3, adbg2 = _adaptive_de00_pool(
                        hue_cand,
                        anchor_lab=tuple(map(float, anchor_eff)),
                        thr_base=float(thr_used),
                        min_n=int(_POOL_FALLBACK_MIN_N),
                        cap=float(_POOL_FALLBACK_DE00_CAP),
                    )
                    if len(df_pool3) > 0:
                        df_pool2 = df_pool3
                        if debug:
                            print("  FAILSAFE: hue-band subset used", hdbg, adbg2)

            df_pool = df_pool2
            if pool_cap > 0 and len(df_pool) > pool_cap:
                df_pool = df_pool.head(pool_cap).copy()

            if debug:
                print("  FAILSAFE: adaptive ΔE used", adbg)

            if df_pool.empty:
                return df_pool

        # UPDATED FAILSAFE (coherent gate):
        # Use naming-prob pool when effective support is weak (not only 0).
        req_label = _first_color_token_from_nlp(nlp_res)
        use_naming_pool = False
        if req_label and _naming_prob_label_supported(req_label, inv_with_probs=inv_eff):
            if (n_support_eff < int(_NAMING_POOL_MIN_N)) or (float(min_de_eff) > float(thr_used)):
                use_naming_pool = True
            if len(df_pool) < int(_NAMING_POOL_MIN_N):
                use_naming_pool = True

        if use_naming_pool:
            name_pool, ndbg = _pool_by_naming_prob(
                inv_eff,
                label=req_label,
                pool_topn=int(_NAMING_POOL_TOPN),
                p_min=float(_NAMING_POOL_P_MIN),
                min_n=int(_NAMING_POOL_MIN_N),
            )
            if not name_pool.empty:
                df_pool = name_pool
                if pool_cap > 0 and len(df_pool) > pool_cap:
                    df_pool = df_pool.head(pool_cap).copy()
                if debug:
                    ndbg = dict(ndbg)
                    ndbg["gate"] = {
                        "n_support_eff": int(n_support_eff),
                        "min_de_eff": float(min_de_eff),
                        "thr_used": float(thr_used),
                        "pool_n_before": int(len(name_pool)),
                    }
                    print("  FAILSAFE: naming-prob pool used", ndbg)

    else:
        df_pool = inv.head(pool_cap).copy()
        if debug:
            print("\n[PHASE A]")
            print(f"  text={text!r}")
            print(f"  has_color={has_color}  anchor_lab={anchor_lab}")
            print(f"  pool_cap={pool_cap}  pool_n={len(df_pool)}  inv_n={len(inv)}")

    # ==========================
    # PHASE B — scoring
    # ==========================
    nlp_constraints = tuple(_get(nlp_res, "constraints", ()) or ())
    query = QuerySpec(anchor_lab=anchor_eff, constraints=nlp_constraints)

    if debug and nlp_constraints:
        print("\n[PHASE B][CONSTRAINTS]")
        for c in nlp_constraints:
            axis = _get_enum_value(_get(c, "axis", None))
            direction = _get_enum_value(_get(c, "direction", None))
            strength = _get_enum_value(_get(c, "strength", None))
            evidence = _get(c, "evidence", None)
            meta = _get(c, "meta", None)
            print(f"  axis={axis} dir={direction} strength={strength} evidence={evidence!r} meta={meta}")

    df_pool = _attach_naming_probs(df_pool)

    # within-family constraints require: label in distributions + distributions loaded
    family_label_for_constraints = family_label if (family_label is not None and family_in_dists and bool(dists)) else None
    family_specs = _family_specs_from_nlp(nlp_constraints) if (family_label_for_constraints is not None) else []

    if debug:
        print("\n[PHASE B][FAMILY-CONSTRAINTS]")
        print(f"  family_label={family_label!r}  in_dists={family_in_dists}")
        print(f"  specs_n={len(family_specs)}")
        if family_label is not None:
            pcol = f"p_{family_label}"
            print(
                f"  pcol_present={pcol in df_pool.columns}  "
                f"p_min={_FAMILY_P_MIN}  p_hi={_FAMILY_P_HI}  floor={_FAMILY_FLOOR}"
            )
        if family_label_for_constraints is None:
            print("  within_family_applied=False")
        else:
            print("  within_family_applied=True")
            if family_specs:
                for s in family_specs:
                    print(f"   - {s.axis}:{s.direction} q_lo={s.q_lo} q_hi={s.q_hi} strength={s.strength}")

    scored = score_shades(
        df_pool,
        query,
        lambda_constraints=float(lambda_constraints),
        lambda_preference=float(lambda_preference),
        calibration=assets.calibration,
        preference_weights=getattr(assets, "preference_weights", None),
        # within-family (only applied if label+specs exist AND label_distributions supports it)
        constraint_label=family_label_for_constraints,
        constraint_specs=family_specs,
        label_distributions=dists if (family_label_for_constraints is not None and dists) else None,
        constraint_p_min=float(_FAMILY_P_MIN),
        constraint_p_hi=float(_FAMILY_P_HI),
        constraint_floor=float(_FAMILY_FLOOR),
    )

    scored = _restore_lab_cols(df_pool, scored)

    score_col = "score_total" if "score_total" in scored.columns else ("score" if "score" in scored.columns else None)
    if score_col is None:
        return scored.head(int(topk)).copy()

    return scored.sort_values(score_col, ascending=False).head(int(topk)).copy()


__all__ = ["recommend_from_text", "resolve_anchor_from_text", "resolve_effective_anchor_from_text"]
