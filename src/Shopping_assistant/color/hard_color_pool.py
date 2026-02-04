# src/Shopping_assistant/colors/hard_color_pool.py
"""
Hard color candidate pooling.

Selects candidate shades based on strict color-distance thresholds
and environment-driven parameters prior to scoring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Shopping_assistant.color.deltae import delta_e_ciede2000


@dataclass(frozen=True, slots=True)
class HardColorPoolParams:
    # ΔE00 thresholds
    de00_max: float = 18.0
    de00_max_neutral: float = 12.0

    # Anchor considered "neutral-ish" if chroma below this
    neutral_anchor_c_max: float = 18.0

    # Neutral guard (filters only; never fills)
    neutral_item_min_c: float = 3.0
    neutral_item_max_l: float = 98.5

    # Extra neutral box gate (critical for beige / nude)
    neutral_l_tol: float = 22.0
    neutral_a_tol: float = 10.0
    neutral_b_tol: float = 14.0


def params_from_env() -> HardColorPoolParams:
    def _f(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except Exception:
            return float(default)

    return HardColorPoolParams(
        de00_max=_f("SA_COLOR_PHASEA_DE00_MAX", 18.0),
        de00_max_neutral=_f("SA_COLOR_PHASEA_DE00_MAX_NEUTRAL", 12.0),
        neutral_anchor_c_max=_f("SA_COLOR_PHASEA_NEUTRAL_ANCHOR_C_MAX", 18.0),
        neutral_item_min_c=_f("SA_COLOR_PHASEA_NEUTRAL_ITEM_MIN_C", 3.0),
        neutral_item_max_l=_f("SA_COLOR_PHASEA_NEUTRAL_ITEM_MAX_L", 98.5),
        neutral_l_tol=_f("SA_COLOR_PHASEA_NEUTRAL_L_TOL", 22.0),
        neutral_a_tol=_f("SA_COLOR_PHASEA_NEUTRAL_A_TOL", 10.0),
        neutral_b_tol=_f("SA_COLOR_PHASEA_NEUTRAL_B_TOL", 14.0),
    )


def hard_color_pool(
    inv: pd.DataFrame,
    *,
    anchor_lab: tuple[float, float, float] | None,
    params: HardColorPoolParams | None = None,
) -> pd.DataFrame:
    """
    Phase A (STRICT):
      - Compute ΔE00(anchor, item) as _de00_anchor
      - Keep ONLY items with _de00_anchor <= threshold (no backfill, ever)
      - For neutral-ish anchors (low chroma), apply extra Lab box guards
      - Return sorted by _de00_anchor ascending
    """
    if inv is None or inv.empty:
        return inv.iloc[0:0].copy()
    if anchor_lab is None:
        return inv.iloc[0:0].copy()

    if not {"L_lab", "a_lab", "b_lab"}.issubset(inv.columns):
        return inv.iloc[0:0].copy()

    L0, a0, b0 = map(float, anchor_lab)
    C0 = float(np.hypot(a0, b0))

    p = params or params_from_env()
    neutral_anchor = C0 < float(p.neutral_anchor_c_max)
    thr = float(p.de00_max_neutral if neutral_anchor else p.de00_max)

    L = pd.to_numeric(inv["L_lab"], errors="coerce").to_numpy(float)
    a = pd.to_numeric(inv["a_lab"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(inv["b_lab"], errors="coerce").to_numpy(float)

    m = np.isfinite(L) & np.isfinite(a) & np.isfinite(b)
    d = np.full(len(inv), np.inf, dtype=float)
    if int(m.sum()) > 0:
        d[m] = delta_e_ciede2000(L[m], a[m], b[m], L0, a0, b0)

    out = inv.copy()
    out["_de00_anchor"] = d

    # strict ΔE00 threshold
    out = out[np.isfinite(out["_de00_anchor"]) & (out["_de00_anchor"] <= thr)].copy()
    if out.empty:
        return out

    # neutral-only guards
    if neutral_anchor:
        L2 = pd.to_numeric(out["L_lab"], errors="coerce").to_numpy(float)
        a2 = pd.to_numeric(out["a_lab"], errors="coerce").to_numpy(float)
        b2 = pd.to_numeric(out["b_lab"], errors="coerce").to_numpy(float)
        C2 = np.hypot(a2, b2)

        mm = np.isfinite(L2) & np.isfinite(a2) & np.isfinite(b2) & np.isfinite(C2)
        keep = np.ones(len(out), dtype=bool)

        # drop near-white / near-clear
        keep &= ~(mm & (L2 >= float(p.neutral_item_max_l)))
        keep &= ~(mm & (C2 <= float(p.neutral_item_min_c)))

        # critical: Lab box gate around neutral anchor
        keep &= ~(mm & (np.abs(L2 - L0) > float(p.neutral_l_tol)))
        keep &= ~(mm & (np.abs(a2 - a0) > float(p.neutral_a_tol)))
        keep &= ~(mm & (np.abs(b2 - b0) > float(p.neutral_b_tol)))

        out = out.loc[keep].copy()
        if out.empty:
            return out

    return out.sort_values("_de00_anchor", ascending=True)
