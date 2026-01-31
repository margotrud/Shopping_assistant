# Scripts/build_domain_color_anchors.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "nlp" / "domain_color_anchors.json"

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


# -----------------------------
# Color math: hex -> Lab (D65) + hue
# -----------------------------
def _hex_to_rgb01(h: str) -> Optional[Tuple[float, float, float]]:
    h = str(h or "").strip()
    if not h:
        return None
    if not h.startswith("#"):
        h = "#" + h
    if not HEX_RE.match(h):
        return None
    r = int(h[1:3], 16) / 255.0
    g = int(h[3:5], 16) / 255.0
    b = int(h[5:7], 16) / 255.0
    return r, g, b


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _rgb01_to_xyz(r: float, g: float, b: float) -> Tuple[float, float, float]:
    r, g, b = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return X, Y, Z


def _xyz_to_lab(X: float, Y: float, Z: float) -> Tuple[float, float, float]:
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = X / Xn, Y / Yn, Z / Zn
    d = 6 / 29

    def f(t: float) -> float:
        return t ** (1 / 3) if t > d**3 else (t / (3 * d**2) + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return float(L), float(a), float(b)


def _hex_to_lab(h: str) -> Optional[Tuple[float, float, float]]:
    rgb = _hex_to_rgb01(h)
    if rgb is None:
        return None
    X, Y, Z = _rgb01_to_xyz(*rgb)
    return _xyz_to_lab(X, Y, Z)


def _lab_C(lab: Tuple[float, float, float]) -> float:
    _, a, b = lab
    return float((a * a + b * b) ** 0.5)


def _hex_hue_deg(h: str) -> Optional[float]:
    rgb = _hex_to_rgb01(h)
    if rgb is None:
        return None
    r, g, b = rgb
    mx, mn = max(r, g, b), min(r, g, b)
    if mx == mn:
        return 0.0
    d = mx - mn
    if mx == r:
        h_ = ((g - b) / d) % 6
    elif mx == g:
        h_ = (b - r) / d + 2
    else:
        h_ = (r - g) / d + 4
    return float((h_ * 60.0) % 360.0)


def _circ_dist_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _quantile(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _circ_mean_deg(vals: np.ndarray) -> Optional[float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    ang = np.deg2rad(vals.astype(np.float64))
    s = float(np.mean(np.sin(ang)))
    c = float(np.mean(np.cos(ang)))
    if (s == 0.0) and (c == 0.0):
        return None
    m = np.arctan2(s, c)
    return float(np.rad2deg(m) % 360.0)


def _lab_medoid_hex(hexes: List[str]) -> Optional[str]:
    labs = []
    for hx in hexes:
        lab = _hex_to_lab(hx)
        if lab is None:
            continue
        labs.append((hx, np.array(lab, dtype=np.float64)))
    if not labs:
        return None

    X = np.stack([v for _, v in labs], axis=0)  # (n,3)
    x2 = np.sum(X * X, axis=1, keepdims=True)
    D = x2 + x2.T - 2.0 * (X @ X.T)
    np.fill_diagonal(D, 0.0)
    s = np.sum(D, axis=1)
    i = int(np.argmin(s))
    return labs[i][0]


def _word_boundary_pat(token: str) -> re.Pattern:
    t = re.escape(token.strip().lower())
    return re.compile(rf"\b{t}\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Params:
    # Hue band
    hue_window_deg: float = float(os.environ.get("SA_DOMAIN_ANCHOR_HUE_WINDOW_DEG", "22"))
    hue_window_max_deg: float = float(os.environ.get("SA_DOMAIN_ANCHOR_HUE_WINDOW_MAX_DEG", "55"))
    hue_window_step_deg: float = float(os.environ.get("SA_DOMAIN_ANCHOR_HUE_WINDOW_STEP_DEG", "6"))

    # Minimum data
    min_n_pool: int = int(os.environ.get("SA_DOMAIN_ANCHOR_MIN_N_POOL", "80"))
    min_n_kept: int = int(os.environ.get("SA_DOMAIN_ANCHOR_MIN_N_KEPT", "30"))

    # Domain hue_ref estimation using text matches (preferred)
    min_n_text_match: int = int(os.environ.get("SA_DOMAIN_ANCHOR_MIN_N_TEXT_MATCH", "35"))

    # Absolute clamps
    L_abs_min: float = float(os.environ.get("SA_DOMAIN_ANCHOR_L_ABS_MIN", "45.0"))
    L_abs_max: float = float(os.environ.get("SA_DOMAIN_ANCHOR_L_ABS_MAX", "78.0"))
    C_abs_min: float = float(os.environ.get("SA_DOMAIN_ANCHOR_C_ABS_MIN", "10.0"))
    C_abs_max: float = float(os.environ.get("SA_DOMAIN_ANCHOR_C_ABS_MAX", "55.0"))

    # Quantile window inside hue band
    L_qlo: float = float(os.environ.get("SA_DOMAIN_ANCHOR_L_QLO", "0.35"))
    L_qhi: float = float(os.environ.get("SA_DOMAIN_ANCHOR_L_QHI", "0.85"))
    C_qlo: float = float(os.environ.get("SA_DOMAIN_ANCHOR_C_QLO", "0.20"))
    C_qhi: float = float(os.environ.get("SA_DOMAIN_ANCHOR_C_QHI", "0.55"))

    # NEW: prefer "less vivid" typical chroma within the kept band
    # (generic, avoids per-colors hardcode)
    C_target_q: float = float(os.environ.get("SA_DOMAIN_ANCHOR_C_TARGET_Q", "0.35"))
    C_target_margin: float = float(os.environ.get("SA_DOMAIN_ANCHOR_C_TARGET_MARGIN", "10.0"))

    # NEW: target point selection before medoid
    L_target_q: float = float(os.environ.get("SA_DOMAIN_ANCHOR_L_TARGET_Q", "0.55"))
    select_topk: int = int(os.environ.get("SA_DOMAIN_ANCHOR_SELECT_TOPK", "40"))
    wC: float = float(os.environ.get("SA_DOMAIN_ANCHOR_WC", "0.90"))
    wC_abs: float = float(os.environ.get("SA_DOMAIN_ANCHOR_WC_ABS", "0.12"))


def _filter_neutral(sub: pd.DataFrame, P: Params, relax: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    relax=0 strict, 1 medium, 2 loose. Returns (filtered_df, debug_meta)
    """
    if relax == 0:
        Lmin, Lmax = P.L_abs_min, P.L_abs_max
        Cmin, Cmax = P.C_abs_min, P.C_abs_max
        qlo_L, qhi_L, qlo_C, qhi_C = P.L_qlo, P.L_qhi, P.C_qlo, P.C_qhi
        c_margin = P.C_target_margin
    elif relax == 1:
        Lmin, Lmax = P.L_abs_min - 5.0, P.L_abs_max
        Cmin, Cmax = P.C_abs_min, P.C_abs_max + 10.0
        qlo_L, qhi_L = P.L_qlo, P.L_qhi
        qlo_C, qhi_C = P.C_qlo, min(0.65, P.C_qhi + 0.10)
        c_margin = P.C_target_margin + 6.0
    else:
        Lmin, Lmax = P.L_abs_min - 8.0, P.L_abs_max
        Cmin, Cmax = P.C_abs_min, P.C_abs_max + 18.0
        qlo_L, qhi_L = P.L_qlo, P.L_qhi
        qlo_C, qhi_C = P.C_qlo, min(0.75, P.C_qhi + 0.20)
        c_margin = P.C_target_margin + 10.0

    sub0 = sub[(sub["L"] >= Lmin) & (sub["L"] <= Lmax) & (sub["C"] >= Cmin) & (sub["C"] <= Cmax)].copy()

    meta: Dict[str, Any] = {
        "abs": {"Lmin": float(Lmin), "Lmax": float(Lmax), "Cmin": float(Cmin), "Cmax": float(Cmax)},
        "q": {"L_qlo": float(qlo_L), "L_qhi": float(qhi_L), "C_qlo": float(qlo_C), "C_qhi": float(qhi_C)},
        "target": {"C_target_q": float(P.C_target_q), "C_target_margin": float(c_margin)},
        "L": {},
        "C": {},
    }

    if len(sub0) == 0:
        return sub0, meta

    L = sub0["L"].to_numpy(dtype=np.float64)
    C = sub0["C"].to_numpy(dtype=np.float64)

    L_lo, L_hi = _quantile(L, qlo_L), _quantile(L, qhi_L)
    C_lo, C_hi_q = _quantile(C, qlo_C), _quantile(C, qhi_C)

    # NEW: cap chroma upper bound toward a "typical" lower quantile (+margin)
    C_t = _quantile(C, float(P.C_target_q))
    C_hi = min(float(C_hi_q), float(C_t + c_margin))

    meta["L"] = {"lo": float(L_lo), "hi": float(L_hi)}
    meta["C"] = {"lo": float(C_lo), "hi_q": float(C_hi_q), "C_t": float(C_t), "hi": float(C_hi)}

    sub1 = sub0[(sub0["L"] >= L_lo) & (sub0["L"] <= L_hi) & (sub0["C"] >= C_lo) & (sub0["C"] <= C_hi)].copy()
    return sub1, meta


def _choose_anchor_from_kept(sub_kept: pd.DataFrame, P: Params) -> Optional[str]:
    """
    NEW:
    - compute target (L_t, C_t) inside kept band
    - score points by distance to target + mild penalty on absolute chroma
    - take topk, then medoid => stable, "typical", less vivid
    """
    if len(sub_kept) == 0:
        return None

    L = sub_kept["L"].to_numpy(dtype=np.float64)
    C = sub_kept["C"].to_numpy(dtype=np.float64)

    L_t = _quantile(L, float(P.L_target_q))
    C_t = _quantile(C, float(P.C_target_q))

    # score: close to target, plus small absolute chroma penalty to avoid vivid tail
    score = np.abs(L - L_t) + float(P.wC) * np.abs(C - C_t) + float(P.wC_abs) * C

    topk = int(max(10, min(int(P.select_topk), len(sub_kept))))
    idx = np.argpartition(score, kth=topk - 1)[:topk]
    shortlist = sub_kept.iloc[idx]
    hx = _lab_medoid_hex(shortlist["chip_hex"].tolist())
    return hx


def main() -> int:
    enriched = os.environ.get("SA_ENRICHED_CSV_PATH", "")
    if not enriched:
        raise RuntimeError("SA_ENRICHED_CSV_PATH is not set.")

    inv = pd.read_csv(enriched)
    if "chip_hex" not in inv.columns:
        raise RuntimeError("inventory missing chip_hex column.")

    inv["chip_hex"] = inv["chip_hex"].astype(str).str.lower().str.strip()
    inv = inv[inv["chip_hex"].str.match(HEX_RE, na=False)].copy()

    pnm = inv["product_name"].astype(str) if "product_name" in inv.columns else ""
    snm = inv["shade_name"].astype(str) if "shade_name" in inv.columns else ""
    inv["_txt"] = (pnm + " " + snm).astype(str).str.lower()

    labs = inv["chip_hex"].map(_hex_to_lab)
    inv["L"] = labs.map(lambda x: x[0] if x else np.nan)
    inv["a"] = labs.map(lambda x: x[1] if x else np.nan)
    inv["b"] = labs.map(lambda x: x[2] if x else np.nan)
    inv["C"] = labs.map(lambda x: _lab_C(x) if x else np.nan)
    inv["hue"] = inv["chip_hex"].map(_hex_hue_deg)

    inv = inv[np.isfinite(inv["L"]) & np.isfinite(inv["C"]) & np.isfinite(inv["hue"])].copy()

    # base lexicon keys
    from Shopping_assistant.nlp.runtime.lexicon import ColorLexicon

    lex = ColorLexicon.load()
    base_keys = sorted([k for k in lex.raw_index.keys() if " " not in k])

    P = Params()
    out: Dict[str, Any] = {"meta": {"params": P.__dict__}, "anchors": {}}

    for k in base_keys:
        info = lex.raw_index.get(k) or {}
        hx0 = str(info.get("hex") or "")
        hue_lex = _hex_hue_deg(hx0)
        if hue_lex is None:
            continue

        # domain-first hue_ref via text matches
        pat = _word_boundary_pat(k)
        m = inv["_txt"].str.contains(pat, na=False)
        hue_ref_src = "lexicon"
        hue_ref = float(hue_lex)

        n_text = int(m.sum())
        if n_text >= P.min_n_text_match:
            hue_m = inv.loc[m, "hue"].to_numpy(dtype=np.float64)
            hm = _circ_mean_deg(hue_m)
            if hm is not None:
                hue_ref = float(hm)
                hue_ref_src = "inventory-text"

        # adaptive hue window
        window = float(P.hue_window_deg)
        sub = None

        while True:
            d = inv["hue"].map(lambda x: _circ_dist_deg(float(x), float(hue_ref)))
            cand = inv[d <= window].copy()
            if len(cand) >= P.min_n_pool:
                sub = cand
                break
            if window + P.hue_window_step_deg > P.hue_window_max_deg:
                sub = None
                break
            window += float(P.hue_window_step_deg)

        if sub is None or len(sub) < P.min_n_pool:
            continue

        # adaptive neutral filtering
        sub2 = None
        relax_used = None
        L_filters = None
        C_filters = None

        for relax in (0, 1, 2):
            cand2, meta = _filter_neutral(sub, P, relax=relax)
            if len(cand2) >= P.min_n_kept:
                sub2 = cand2
                relax_used = int(relax)
                L_filters = {
                    "abs_min": float(meta["abs"]["Lmin"]),
                    "abs_max": float(meta["abs"]["Lmax"]),
                    "qlo": float(meta["q"]["L_qlo"]),
                    "qhi": float(meta["q"]["L_qhi"]),
                    "lo": float(meta["L"].get("lo")) if isinstance(meta.get("L"), dict) else None,
                    "hi": float(meta["L"].get("hi")) if isinstance(meta.get("L"), dict) else None,
                }
                C_filters = {
                    "abs_min": float(meta["abs"]["Cmin"]),
                    "abs_max": float(meta["abs"]["Cmax"]),
                    "qlo": float(meta["q"]["C_qlo"]),
                    "qhi": float(meta["q"]["C_qhi"]),
                    "lo": float(meta["C"].get("lo")) if isinstance(meta.get("C"), dict) else None,
                    "hi_q": float(meta["C"].get("hi_q")) if isinstance(meta.get("C"), dict) else None,
                    "C_t": float(meta["C"].get("C_t")) if isinstance(meta.get("C"), dict) else None,
                    "hi": float(meta["C"].get("hi")) if isinstance(meta.get("C"), dict) else None,
                }
                break

        if sub2 is None:
            continue

        hx = _choose_anchor_from_kept(sub2, P)
        if not hx:
            continue

        lab = _hex_to_lab(hx)
        if lab is None:
            continue

        out["anchors"][k] = {
            "hex": hx,
            "hue_ref": float(hue_ref),
            "hue_ref_src": hue_ref_src,
            "hue_window_deg_used": float(window),
            "n_text_match": int(n_text),
            "n_pool": int(len(sub)),
            "n_kept": int(len(sub2)),
            "relax_used": int(relax_used) if relax_used is not None else None,
            "L_anchor": float(lab[0]),
            "C_anchor": float(_lab_C(lab)),
            "policy": "inventory-domain-neutral-typical-medoid",
            "L_filters": L_filters,
            "C_filters": C_filters,
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build_domain_color_anchors] wrote -> {OUT_PATH} (n={len(out['anchors'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
