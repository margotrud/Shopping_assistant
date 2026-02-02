# src/Shopping_assistant/reco/_domain_anchor.py
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pooling import _ensure_hue_rgb_deg
from ._colorconv import _hex_to_hsv_h_deg, _circular_abs_diff_deg
from ._constants import (
    _DOMAIN_ANCHOR_MIN_N,
    _DOMAIN_ANCHOR_TOPN,
    _DOMAIN_ANCHOR_FIX_LABELS,
    _DOMAIN_ANCHOR_HUE_BAND_DEG,
    _DOMAIN_ANCHOR_HUE_TRIM_Q,
)


def _domain_anchor_from_naming_probs(
    inv: pd.DataFrame,
    *,
    label: str,
    p_min: float,
    anchor_lab_lexicon: tuple[float, float, float] | None = None,  # NEW
    anchor_hex: str | None = None,  # NEW (optional)
) -> tuple[float, float, float] | None:
    if inv is None or inv.empty:
        return None
    if not {"L_lab", "a_lab", "b_lab", "chip_hex"}.issubset(inv.columns):
        return None

    label_l = str(label).lower()
    pcol = f"p_{label_l}"
    if pcol not in inv.columns:
        return None

    w_all = pd.to_numeric(inv[pcol], errors="coerce").fillna(0.0).to_numpy(float)

    m = w_all >= float(p_min)
    if int(np.sum(m)) < int(_DOMAIN_ANCHOR_MIN_N):
        idx = np.argsort(-w_all)
        idx = idx[w_all[idx] > 0.0][: int(_DOMAIN_ANCHOR_TOPN)]
        if len(idx) < int(_DOMAIN_ANCHOR_MIN_N):
            return None
        sub = inv.iloc[idx].copy()
        w = w_all[idx]
    else:
        sub = inv.loc[m].copy()
        w = w_all[m]

    # ---------- DEFAULT PATH (unchanged) ----------
    if label_l not in _DOMAIN_ANCHOR_FIX_LABELS:
        L = pd.to_numeric(sub["L_lab"], errors="coerce").to_numpy(float)
        a = pd.to_numeric(sub["a_lab"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(sub["b_lab"], errors="coerce").to_numpy(float)
        ok = np.isfinite(L) & np.isfinite(a) & np.isfinite(b) & np.isfinite(w) & (w > 0.0)
        if not np.any(ok):
            return None
        ww = w[ok]
        ww = ww / (np.sum(ww) + 1e-12)
        return (float(np.sum(L[ok] * ww)), float(np.sum(a[ok] * ww)), float(np.sum(b[ok] * ww)))

    # ---------- FIXED PATH (ONLY purple/violet/brown) ----------
    # Use HSV hue computed from chip_hex (not Lab-hue) for band selection.
    sub2 = _ensure_hue_rgb_deg(sub)
    h = pd.to_numeric(sub2.get("_H_rgb_deg", pd.Series([np.nan] * len(sub2))), errors="coerce").to_numpy(float)

    ok_h = np.isfinite(h) & np.isfinite(w) & (w > 0.0)
    if not np.any(ok_h):
        return None

    ok_idx = np.where(ok_h)[0]
    sub_ok = sub2.iloc[ok_idx].copy()
    h_sub = h[ok_h].astype(float)
    w_ok = w[ok_h].astype(float)

    # reference hue:
    #   1) anchor_hex HSV hue if available
    #   2) hue of max-weight candidate (stable fallback)
    h0 = _hex_to_hsv_h_deg(anchor_hex) if isinstance(anchor_hex, str) else None
    if h0 is None:
        j = int(np.argmax(w_ok))
        hx = str(sub_ok.iloc[j].get("chip_hex", ""))
        h0 = _hex_to_hsv_h_deg(hx) if hx else None
    if h0 is None:
        return None

    hd = _circular_abs_diff_deg(h_sub, float(h0))
    band = float(_DOMAIN_ANCHOR_HUE_BAND_DEG)

    keep = hd <= band
    if int(np.sum(keep)) < int(_DOMAIN_ANCHOR_MIN_N):
        ord_idx = np.argsort(hd)
        ord_idx = ord_idx[: max(int(_DOMAIN_ANCHOR_MIN_N), 1)]
        sub_ok = sub_ok.iloc[ord_idx].copy()
        w_ok = w_ok[ord_idx]
        hd = hd[ord_idx]
    else:
        sel = np.where(keep)[0]
        sub_ok = sub_ok.iloc[sel].copy()
        w_ok = w_ok[keep]
        hd = hd[keep]

    # extra trim for brown: remove the farthest hue tail automatically (generic)
    if label_l == "brown":
        q = float(_DOMAIN_ANCHOR_HUE_TRIM_Q)
        thr = float(np.quantile(hd, 1.0 - q)) if len(hd) else np.inf
        keep2 = hd <= thr
        if int(np.sum(keep2)) >= int(_DOMAIN_ANCHOR_MIN_N):
            sel2 = np.where(keep2)[0]
            sub_ok = sub_ok.iloc[sel2].copy()
            w_ok = w_ok[keep2]
            hd = hd[keep2]

    # reweight by hue distance (generic)
    sigma = max(float(_DOMAIN_ANCHOR_HUE_BAND_DEG) / 2.0, 1e-6)
    w2 = w_ok * np.exp(-((hd / sigma) ** 2))
    if not np.any(np.isfinite(w2)) or float(np.sum(w2)) <= 0.0:
        w2 = w_ok

    L = pd.to_numeric(sub_ok["L_lab"], errors="coerce").to_numpy(float)
    a = pd.to_numeric(sub_ok["a_lab"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(sub_ok["b_lab"], errors="coerce").to_numpy(float)
    ok2 = np.isfinite(L) & np.isfinite(a) & np.isfinite(b) & np.isfinite(w2) & (w2 > 0.0)
    if not np.any(ok2):
        return None
    ww = w2[ok2]
    ww = ww / (np.sum(ww) + 1e-12)

    anch = (
        float(np.sum(L[ok2] * ww)),
        float(np.sum(a[ok2] * ww)),
        float(np.sum(b[ok2] * ww)),
    )
    return anch
