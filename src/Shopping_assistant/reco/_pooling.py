# src/Shopping_assistant/reco/_pooling.py
from __future__ import annotations

import numpy as np
import pandas as pd

from Shopping_assistant.color.deltae import delta_e_ciede2000
from Shopping_assistant.color.hard_color_pool import hard_color_pool, params_from_env

from ._colorconv import _hex_to_hsv_h_deg, _circ_dist_deg, _circular_abs_diff_deg
from ._constants import (
    _HUE_FALLBACK_BAND_DEG,
    _HUE_FALLBACK_MIN_POOL_N,
    _HUE_FALLBACK_CHROMA_Q,
    _HUE_FALLBACK_ANCHOR_C_MIN,
    _POOL_FALLBACK_DE00_STEPS,
    _POOL_FALLBACK_DE00_CAP,
    _POOL_FALLBACK_MIN_N,
    _HUE_FAILSAFE_BAND_DEG,
    _NAMING_POOL_MIN_N,
    _NAMING_POOL_TOPN,
    _NAMING_POOL_P_MIN,
    _DOMAIN_ANCHOR_MIN_N,
    _DOMAIN_ANCHOR_TOPN,
    _DOMAIN_ANCHOR_FIX_LABELS,
    _DOMAIN_ANCHOR_HUE_BAND_DEG,
    _DOMAIN_ANCHOR_HUE_TRIM_Q,
)
from ._naming_probs import _attach_naming_probs, _naming_prob_label_supported


def _restore_feature_cols(df_pool: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """
    Re-attach feature columns that tests/downstream code expect.
    score_shades may drop input columns; we merge back deterministically.
    """
    if scored is None or scored.empty or df_pool is None or df_pool.empty:
        return scored

    # Candidate keys (stable join)
    key_candidates = [("product_id", "shade_id"), ("shade_id",), ("product_id",)]
    keys = None
    for kc in key_candidates:
        if all(k in df_pool.columns for k in kc) and all(k in scored.columns for k in kc):
            keys = list(kc)
            break

    # Columns to restore (only if present in df_pool and missing in scored)
    want_cols = (
        # identity-ish
        "chip_hex",
        # RGB/HSL features (your failing test needs light_hsl)
        "r",
        "g",
        "b",
        "h_hsl",
        "s_hsl",
        "light_hsl",
        # Lab features
        "L_lab",
        "a_lab",
        "b_lab",
        "C_lab",
        "H_lab_deg",
        # debug
        "_de00_anchor",
        "_H_rgb_deg",
    )

    want = [c for c in want_cols if (c in df_pool.columns and c not in scored.columns)]
    if not want:
        return scored

    if keys is None:
        # Index-aligned restore only if safe
        if df_pool.index.equals(scored.index):
            out = scored.copy()
            for c in want:
                out[c] = df_pool[c]
            return out
        return scored

    left = scored.copy()
    right = df_pool[keys + want].drop_duplicates(subset=keys).copy()

    # normalize key dtypes (avoid merge misses)
    for k in keys:
        left[k] = left[k].astype(str)
        right[k] = right[k].astype(str)

    return left.merge(right, on=keys, how="left")


def _robust_medoid_lab(sub: pd.DataFrame, w: np.ndarray) -> tuple[float, float, float] | None:
    # pick the point minimizing weighted squared distance in (a,b) primarily (cheap + stable)
    if sub is None or sub.empty:
        return None
    L = pd.to_numeric(sub["L_lab"], errors="coerce").to_numpy(float)
    a = pd.to_numeric(sub["a_lab"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(sub["b_lab"], errors="coerce").to_numpy(float)
    ok = np.isfinite(L) & np.isfinite(a) & np.isfinite(b) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return None
    L, a, b, w = L[ok], a[ok], b[ok], w[ok]
    # normalize weights
    w = w / (np.sum(w) + 1e-12)

    # medoid in (a,b) space (fast O(n^2) ok for <=250)
    A = np.vstack([a, b]).T
    # squared distances
    D = np.sum((A[:, None, :] - A[None, :, :]) ** 2, axis=2)
    # weighted sum of distances from each candidate to all points
    cost = D @ w
    i = int(np.argmin(cost))
    return (float(L[i]), float(a[i]), float(b[i]))


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


def _ensure_de00_anchor_col(df_pool: pd.DataFrame, anchor_lab: tuple[float, float, float]) -> pd.DataFrame:
    """
    Ensure df_pool has '_de00_anchor' for debugging/inspection, regardless of pool builder.
    """
    if df_pool is None or df_pool.empty:
        return df_pool
    if "_de00_anchor" in df_pool.columns:
        return df_pool
    if not {"L_lab", "a_lab", "b_lab"}.issubset(df_pool.columns):
        return df_pool
    d = _de00_to_anchor(df_pool, anchor_lab)
    out = df_pool.copy()
    out["_de00_anchor"] = d
    return out


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
