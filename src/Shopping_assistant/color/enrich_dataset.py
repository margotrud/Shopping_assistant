# src/Shopping_assistant/color/enrich_dataset.py
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# IO / Paths
# ---------------------------------------------------------------------

_RGB_RE = re.compile(r"^\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*$")


def _project_root() -> Path:
    # .../src/Shopping_assistant/color/enrich_dataset.py -> project root = parents[3]
    return Path(__file__).resolve().parents[3]


def _default_infile() -> Path:
    return _project_root() / "data" / "Sephora_lipsticks_raw_items_with_chip_rgb.csv"


def _default_outdir() -> Path:
    return _project_root() / "data" / "enriched_data"


# ---------------------------------------------------------------------
# Color science: sRGB -> linear RGB -> XYZ (D65) -> Lab -> LCH
# ---------------------------------------------------------------------

# D65 white point (2°)
_D65 = (0.95047, 1.00000, 1.08883)


def _srgb_to_linear(c_srgb_0_1: np.ndarray) -> np.ndarray:
    """
    Vectorized sRGB -> linear RGB (0..1), IEC 61966-2-1
    """
    a = 0.055
    return np.where(
        c_srgb_0_1 <= 0.04045,
        c_srgb_0_1 / 12.92,
        ((c_srgb_0_1 + a) / (1.0 + a)) ** 2.4,
    )


def _linear_rgb_to_xyz(rgb_lin: np.ndarray) -> np.ndarray:
    """
    Vectorized linear RGB -> XYZ with D65, matrix for sRGB.
    rgb_lin: shape (n, 3)
    """
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    return rgb_lin @ M.T


def _f_lab(t: np.ndarray) -> np.ndarray:
    """
    CIE Lab f(t) helper (vectorized)
    """
    delta = 6.0 / 29.0
    return np.where(
        t > delta**3,
        np.cbrt(t),
        (t / (3.0 * delta**2)) + (4.0 / 29.0),
    )


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """
    Vectorized XYZ -> Lab using D65 white point.
    xyz: shape (n, 3)
    returns: shape (n, 3) -> (L*, a*, b*)
    """
    Xn, Yn, Zn = _D65
    x = xyz[:, 0] / Xn
    y = xyz[:, 1] / Yn
    z = xyz[:, 2] / Zn

    fx = _f_lab(x)
    fy = _f_lab(y)
    fz = _f_lab(z)

    L = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=1)


def _lab_to_lch(lab: np.ndarray) -> np.ndarray:
    """
    Vectorized Lab -> LCH(ab):
      L = L*
      C = sqrt(a^2 + b^2)
      H = atan2(b, a) in degrees mapped to [0, 360)
    """
    L = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]
    C = np.sqrt(a * a + b * b)
    H = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
    return np.stack([L, C, H], axis=1)


# ---------------------------------------------------------------------
# sRGB -> HSL (vectorized, 0..1 for s,l, deg for hue)
# ---------------------------------------------------------------------

def _rgb01_to_hsl(rgb01: np.ndarray) -> np.ndarray:
    """
    Vectorized RGB (0..1) -> HSL.
    Returns (hue_deg, sat_hsl, light_hsl)
    """
    r = rgb01[:, 0]
    g = rgb01[:, 1]
    b = rgb01[:, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Lightness
    l = (cmax + cmin) / 2.0

    # Saturation
    s = np.zeros_like(l)
    nonzero = delta > 1e-12
    s[nonzero] = delta[nonzero] / (1.0 - np.abs(2.0 * l[nonzero] - 1.0))

    # Hue
    h = np.zeros_like(l)
    # avoid division by zero
    d = np.where(nonzero, delta, 1.0)

    rmax = (cmax == r) & nonzero
    gmax = (cmax == g) & nonzero
    bmax = (cmax == b) & nonzero

    h[rmax] = ((g[rmax] - b[rmax]) / d[rmax]) % 6.0
    h[gmax] = ((b[gmax] - r[gmax]) / d[gmax]) + 2.0
    h[bmax] = ((r[bmax] - g[bmax]) / d[bmax]) + 4.0

    hue_deg = (h * 60.0) % 360.0
    return np.stack([hue_deg, s, l], axis=1)


# ---------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------

def _colorfulness_hasler_susstrunk(rgb01: np.ndarray) -> np.ndarray:
    """
    Hasler–Süsstrunk colorfulness (approx, using sRGB 0..1).
    Ref formula:
      rg = R - G
      yb = 0.5*(R + G) - B
      C = sqrt(std(rg)^2 + std(yb)^2) + 0.3*sqrt(mean(rg)^2 + mean(yb)^2)
    For single colors, std=0, so this reduces to 0.3*sqrt(rg^2 + yb^2).
    This is still a useful monotonic feature.
    """
    r = rgb01[:, 0]
    g = rgb01[:, 1]
    b = rgb01[:, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return 0.3 * np.sqrt(rg * rg + yb * yb)


def _safe_clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# ---------------------------------------------------------------------
# Parsing / Validation
# ---------------------------------------------------------------------

def _parse_chip_rgb_series(chip_rgb: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Parse 'chip_rgb' formatted like "166,68,73" into r,g,b numeric columns.
    Missing/invalid -> NaN.
    """
    s = chip_rgb.astype(str).str.strip()
    m = s.str.extract(_RGB_RE)
    r = pd.to_numeric(m[0], errors="coerce")
    g = pd.to_numeric(m[1], errors="coerce")
    b = pd.to_numeric(m[2], errors="coerce")

    # clamp out-of-range to NaN (data quality)
    for col in (r, g, b):
        col[(col < 0) | (col > 255)] = np.nan
    return r, g, b


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class EnrichConfig:
    rgb_col: str = "chip_rgb"
    drop_rows_missing_rgb: bool = False
    out_suffix: str = "_enriched"


def enrich_dataframe(df: pd.DataFrame, *, cfg: EnrichConfig = EnrichConfig()) -> pd.DataFrame:
    if cfg.rgb_col not in df.columns:
        raise KeyError(f"Missing required column '{cfg.rgb_col}'")

    r, g, b = _parse_chip_rgb_series(df[cfg.rgb_col])
    out = df.copy()

    out["r"] = r
    out["g"] = g
    out["b"] = b

    if cfg.drop_rows_missing_rgb:
        out = out.dropna(subset=["r", "g", "b"]).copy()

    # Prepare arrays
    rgb255 = out[["r", "g", "b"]].to_numpy(dtype=np.float64)
    rgb01 = rgb255 / 255.0
    rgb01 = _safe_clip01(rgb01)

    rgb_lin = _srgb_to_linear(rgb01)
    xyz = _linear_rgb_to_xyz(rgb_lin)
    lab = _xyz_to_lab(xyz)
    lch = _lab_to_lch(lab)
    hsl = _rgb01_to_hsl(rgb01)

    # Core perceptual features
    out["L_lab"] = lab[:, 0]
    out["a_lab"] = lab[:, 1]
    out["b_lab"] = lab[:, 2]

    out["C_lab"] = lch[:, 1]
    out["H_lab_deg"] = lch[:, 2]  # perceptual hue angle (Lab)

    # HSL (useful for intuition / UI)
    out["hue_hsl_deg"] = hsl[:, 0]
    out["sat_hsl"] = hsl[:, 1]
    out["light_hsl"] = hsl[:, 2]

    # Relative luminance Y (linear RGB) in [0..1] with sRGB coefficients
    # NOTE: This is the Y of XYZ when using sRGB->XYZ matrix; xyz[:,1] is already Y in [0..1]-ish.
    out["Y_rel"] = xyz[:, 1]

    # Derived ML-friendly features
    out["depth"] = 100.0 - out["L_lab"]  # higher = darker/deeper
    out["warmth"] = out["b_lab"] - 0.5 * out["a_lab"]
    out["sat_eff"] = out["C_lab"] / (out["L_lab"].clip(lower=1e-6))  # avoid div0

    # Colorfulness (monotonic per-color proxy)
    out["colorfulness"] = _colorfulness_hasler_susstrunk(rgb01)

    # Sanity cleanup: if RGB missing, computed values are garbage -> set NaN consistently
    missing_rgb = out[["r", "g", "b"]].isna().any(axis=1)
    computed_cols = [
        "L_lab",
        "a_lab",
        "b_lab",
        "C_lab",
        "H_lab_deg",
        "hue_hsl_deg",
        "sat_hsl",
        "light_hsl",
        "Y_rel",
        "depth",
        "warmth",
        "sat_eff",
        "colorfulness",
    ]
    out.loc[missing_rgb, computed_cols] = np.nan

    return out


def enrich_csv(infile: Path, outdir: Path, *, cfg: EnrichConfig = EnrichConfig()) -> Path:
    infile = infile.resolve()
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)
    enriched = enrich_dataframe(df, cfg=cfg)

    outpath = outdir / f"{infile.stem}{cfg.out_suffix}.csv"
    enriched.to_csv(outpath, index=False)
    return outpath


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Enrich Sephora lipstick dataset with perceptual color features from chip_rgb.",
    )
    p.add_argument(
        "--infile",
        type=str,
        default=str(_default_infile()),
        help="Input CSV path (must include chip_rgb column).",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(_default_outdir()),
        help="Output directory for enriched CSV.",
    )
    p.add_argument(
        "--rgb-col",
        type=str,
        default="chip_rgb",
        help="Name of the RGB column (default: chip_rgb).",
    )
    p.add_argument(
        "--drop-missing-rgb",
        action="store_true",
        help="Drop rows with missing/invalid RGB instead of keeping them with NaNs.",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = EnrichConfig(
        rgb_col=args.rgb_col,
        drop_rows_missing_rgb=bool(args.drop_missing_rgb),
    )
    outpath = enrich_csv(Path(args.infile), Path(args.outdir), cfg=cfg)
    print(str(outpath))


if __name__ == "__main__":
    main()
