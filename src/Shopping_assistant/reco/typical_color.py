# src/Shopping_assistant/reco/typical_color.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------- hex -> Lab (D65), CIE76 ----------
def _hex_to_rgb01(h: str) -> Tuple[float, float, float]:
    s = str(h).strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {h!r}")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return r, g, b


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _rgb_to_xyz(r: float, g: float, b: float) -> Tuple[float, float, float]:
    r, g, b = map(_srgb_to_linear, (r, g, b))
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return x, y, z


def _f(t: float) -> float:
    d = 6 / 29
    return t ** (1 / 3) if t > d**3 else (t / (3 * d * d) + 4 / 29)


def _xyz_to_lab(x: float, y: float, z: float) -> Tuple[float, float, float]:
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    fx, fy, fz = _f(x / Xn), _f(y / Yn), _f(z / Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return float(L), float(a), float(b)


def hex_to_lab(hexstr: str) -> Tuple[float, float, float]:
    r, g, b = _hex_to_rgb01(hexstr)
    return _xyz_to_lab(*_rgb_to_xyz(r, g, b))


def _deltaE76(lab: np.ndarray, seed: np.ndarray) -> np.ndarray:
    return np.linalg.norm(lab - seed[None, :], axis=1)


def _safe_hex_to_lab(x: str) -> Tuple[float, float, float] | None:
    try:
        s = str(x).strip()
        if not s:
            return None
        s2 = s.lstrip("#")
        if len(s2) != 6:
            return None
        int(s2, 16)  # validate hex
        return hex_to_lab("#" + s2)
    except Exception:
        return None


# ---------- main: typical median around seed (DEPRECATED) ----------
def apply_typical_median_near_seed(
    df: pd.DataFrame,
    *,
    seed_hex: Optional[str],
    chip_hex_col: str = "chip_hex",
    score_col: str = "score_total",
    neighbor_m: int = 200,  # how many closest-to-seed to define the "family" support
    min_n: int = 30,  # "if many products"
    weight: float = 2.5,  # how strongly to pull toward typical
) -> pd.DataFrame:
    """
    Deprecated behavior (seed-centric).
    This function is kept for backward compatibility but should NOT be used
    to implement "typical center of family" for plain-color queries.

    Current recommended approach:
    - build a family pool in recommend.py
    - compute robust medians inside that pool
    - apply _apply_median_in_pool_rerank() ONLY for plain-color queries
    """
    if df is None or df.empty:
        return df
    if not seed_hex or chip_hex_col not in df.columns:
        return df
    if neighbor_m <= 0 or min_n <= 0 or weight == 0.0:
        return df

    out = df.copy()

    hx = out[chip_hex_col].astype(str).str.strip()
    lab_list = hx.apply(_safe_hex_to_lab)
    ok = lab_list.notna()

    if ok.sum() < min_n:
        return out

    try:
        seed = np.array(hex_to_lab(seed_hex), dtype=float)
    except Exception:
        return out

    lab = np.array(lab_list[ok].tolist(), dtype=float)  # (Nok,3)
    idx_ok = np.flatnonzero(ok.to_numpy())

    d_seed_ok = _deltaE76(lab, seed)

    m = int(min(neighbor_m, len(d_seed_ok)))
    if m < min_n:
        return out

    ord_idx = np.argsort(d_seed_ok)[:m]
    neigh_lab = lab[ord_idx]
    med = np.median(neigh_lab, axis=0)  # median Lab prototype (seed neighborhood)

    d_typ_ok = np.linalg.norm(lab - med[None, :], axis=1)

    out["_dE_seed"] = np.nan
    out["_dE_typical"] = np.nan
    out.loc[out.index[idx_ok], "_dE_seed"] = d_seed_ok
    out.loc[out.index[idx_ok], "_dE_typical"] = d_typ_ok

    score_typ_ok = -d_typ_ok
    out["_score_typical"] = np.nan
    out.loc[out.index[idx_ok], "_score_typical"] = score_typ_ok

    if score_col in out.columns:
        out[score_col] = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0) + float(weight) * out[
            "_score_typical"
        ].fillna(0.0).astype(float)
    else:
        out[score_col] = float(weight) * out["_score_typical"].fillna(0.0).astype(float)

    return out
