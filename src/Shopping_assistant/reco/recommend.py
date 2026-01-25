# src/Shopping_assistant/reco/recommend.py
from __future__ import annotations

import inspect
import re
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from Shopping_assistant.io.assets import AssetBundle, load_default_assets
from Shopping_assistant.nlp import Axis
from Shopping_assistant.nlp.interpretation.preference import interpret_nlp
from Shopping_assistant.nlp.resolve.preference_resolver import resolve_preference
from Shopping_assistant.nlp.resolve.axis_projection import project_axes
from Shopping_assistant.nlp.resolve.axis_merge import merge_axis_intents
from Shopping_assistant.nlp.resolve.axis_thresholds import AxisThreshold, thresholds_from_decisions
from Shopping_assistant.nlp.resolve.scoring_adapter import build_constraints_blob_from_thresholds
from Shopping_assistant.nlp.runtime.lexicon import load_default_lexicon

from Shopping_assistant.color.scoring import (
    Constraint,
    QuerySpec,
    score_shades,
)
from Shopping_assistant.reco.typical_color import apply_typical_median_near_seed

# Keep consistent with Shopping_assistant.color.scoring._parse_constraint()
_CONSTRAINT_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*(<=|>=)\s*((?:low|medium|high|very_high)|(?:[0-9]*\.?[0-9]+))\s*:\s*([0-9]*\.?[0-9]+)\s*$"
)

# Neutral “plain color” regularizer policy:
# - MUST NOT expand candidate pools (keeps hue family)
# - MUST be low-weight (tie-breaker only)
_NEUTRAL_DIMS = ("light_hsl", "sat_hsl")
_NEUTRAL_LEVEL = "medium"
_NEUTRAL_W = 0.02

# Hue-lock neighborhood: use angle(a,b) + chroma + lightness (prevents drift)
_NEIGHBOR_WL = 0.05
_NEIGHBOR_WTHETA = 2500.0  # scale for (rad^2) to compete with Lab numeric ranges
_NEIGHBOR_WC = 0.25

# When we injected neutral-only constraints, clamp constraint influence (tie-breaker)
_NEUTRAL_LAMBDA_MAX = 0.35

# Candidate pool policy (distance-only, no clusters)
_DEFAULT_POOL_TOPN = 800
_MAX_POOL_TOPN = 2000
_POOL_EXPAND_STEP = 400
_MIN_FEASIBLE_MULT = 5  # min_candidates = max(50, _MIN_FEASIBLE_MULT * topk)

# Hue-window pool policy (plain-color mode)
_HUE_WINDOW_DEG_DENSE = 25.0
_HUE_WINDOW_DEG_STEPS = (25.0, 35.0, 45.0, 60.0, 90.0)
_DENSE_REQUIRED_MULT = 5  # dense if n_close_25deg >= _DENSE_REQUIRED_MULT * topk

# Plain-color neutral center pull
_CENTER_DE_LAMBDA = 0.15

# Typical-color policy (DATA-DRIVEN, no hue windows)
_TYPICAL_MIN_N = 30
_TYPICAL_NEIGHBOR_M = 400
_TYPICAL_WEIGHT = 2.5

# Canonical / wearable prior (plain-color only, within locked pool)
_CANONICAL_LAMBDA = 1.0
_CANONICAL_EPS = 1e-6
_CANONICAL_SIGMA_FLOOR = 3.0  # L* and C* are on similar-ish numeric scales here

# Plain-color anchor pull (prevents "purple" -> mauve/berry)
_PLAIN_COLOR_LAMBDA_PREFERENCE_MIN = 0.25
# Plain-color: penalize global chroma extremes (prevents neon reds)
_PLAIN_CHROMA_EXTREME_Q = 0.90
_PLAIN_CHROMA_EXTREME_LAMBDA = 1.25

# Columns we want to preserve in the returned df for explain/debug/UX
_EXPLAIN_COLS = (
    "light_hsl",
    "sat_hsl",
    "warmth",
    "depth",
    "colorfulness",
    "L_lab",
    "C_lab",
    "a_lab",
    "b_lab",
)


def _parse_constraint_token(token: str) -> Constraint:
    raw = token.strip()
    m = _CONSTRAINT_RE.match(raw)
    if not m:
        raise ValueError(f"Invalid constraint token: {raw!r}")

    dim, op, lvl_or_num, w_str = m.groups()
    if lvl_or_num in {"low", "medium", "high", "very_high"}:
        return Constraint(dim=str(dim), op=str(op), level=str(lvl_or_num), cutpoint=None, weight=float(w_str))
    return Constraint(dim=str(dim), op=str(op), level=None, cutpoint=float(lvl_or_num), weight=float(w_str))


# ---------------------------------------------------------------------
# Canonical / wearable prior (data-driven, within pool)
# ---------------------------------------------------------------------


def _robust_center_and_scale(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, float(_CANONICAL_SIGMA_FLOOR)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad
    sigma = max(float(sigma), float(_CANONICAL_SIGMA_FLOOR))
    return med, sigma


def _global_chroma_hi(inv: pd.DataFrame, q: float) -> Optional[float]:
    df = _ensure_C_lab(inv)
    if "C_lab" not in df.columns:
        return None
    vals = pd.to_numeric(df["C_lab"], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.quantile(float(q)))

def _ensure_C_lab(df: pd.DataFrame) -> pd.DataFrame:
    if "C_lab" in df.columns:
        return df
    if {"a_lab", "b_lab"}.issubset(df.columns):
        out = df.copy()
        a = pd.to_numeric(out["a_lab"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(out["b_lab"], errors="coerce").to_numpy(float)
        out["C_lab"] = np.sqrt(a * a + b * b)
        return out
    return df


def _canonical_penalty(df_pool: pd.DataFrame) -> pd.Series:
    df = _ensure_C_lab(df_pool)

    need = {"L_lab", "a_lab", "b_lab", "C_lab"}
    if not need.issubset(df.columns):
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    L = pd.to_numeric(df["L_lab"], errors="coerce").to_numpy(float)
    a = pd.to_numeric(df["a_lab"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(df["b_lab"], errors="coerce").to_numpy(float)
    C = pd.to_numeric(df["C_lab"], errors="coerce").to_numpy(float)

    L_med, L_sig = _robust_center_and_scale(L)
    C_med, C_sig = _robust_center_and_scale(C)
    a_med, _ = _robust_center_and_scale(a)
    b_med, _ = _robust_center_and_scale(b)

    zL2 = ((L - L_med) / (L_sig + _CANONICAL_EPS)) ** 2
    zC2 = ((C - C_med) / (C_sig + _CANONICAL_EPS)) ** 2

    d_typ2 = (L - L_med) ** 2 + (a - a_med) ** 2 + (b - b_med) ** 2
    d_typ2_med = float(np.nanmedian(d_typ2)) if np.isfinite(np.nanmedian(d_typ2)) else 0.0
    d_typ_norm = d_typ2 / (d_typ2_med + _CANONICAL_EPS)

    pen = zL2 + zC2 + 0.02 * d_typ_norm
    pen = np.where(np.isfinite(pen), pen, 0.0)
    return pd.Series(pen, index=df.index, dtype=float)


# ---------------------------------------------------------------------
# Anchor Lab from NLP (NO clusters, NO prototypes required)
# ---------------------------------------------------------------------


def _hex_to_rgb01(hx: str) -> Optional[Tuple[float, float, float]]:
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


def _hex_to_lab(hx: str) -> Optional[Tuple[float, float, float]]:
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


def _get(m, key: str, default=None):
    try:
        if isinstance(m, dict):
            return m.get(key, default)
        return getattr(m, key, default)
    except Exception:
        return default


def _get_enum_value(x) -> Optional[str]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            return x
        return getattr(x, "value", None)
    except Exception:
        return None


def _seed_hex_from_nlp(nlp_res) -> Optional[str]:
    best_hex: Optional[str] = None
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


def _anchor_lab_from_nlp(nlp_res) -> Optional[Tuple[float, float, float]]:
    best_lab: Optional[Tuple[float, float, float]] = None
    best_score = -1.0

    mentions = _get(nlp_res, "mentions", ()) or ()
    try:
        lex = load_default_lexicon()
    except Exception:
        lex = None

    for m in mentions:
        kind = _get_enum_value(_get(m, "kind", None))
        if (kind or "").lower() != "color":
            continue

        pol = _get_enum_value(_get(m, "polarity", None))
        if (pol or "").lower() not in {"like", "neutral", "unknown"}:
            continue

        conf = float(_get(m, "confidence", 0.0) or 0.0)
        meta = _get(m, "meta", {}) or {}

        lab_m: Optional[Tuple[float, float, float]] = None

        L0 = meta.get("lab_L") if isinstance(meta, dict) else None
        a0 = meta.get("lab_a") if isinstance(meta, dict) else None
        b0 = meta.get("lab_b") if isinstance(meta, dict) else None
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


def _has_effective_thresholds(ths: dict[Axis, AxisThreshold] | None) -> bool:
    if not ths:
        return False
    for t in ths.values():
        w = float(getattr(t, "weight", 0.0) or 0.0)
        if w > 0.0:
            return True
    return False


def _has_color_like_mention(nlp_res) -> bool:
    for m in _get(nlp_res, "mentions", ()) or ():
        kind = _get_enum_value(_get(m, "kind", None))
        if (kind or "").lower() != "color":
            continue
        pol = _get_enum_value(_get(m, "polarity", None))
        if (pol or "").lower() in {"like", "neutral", "unknown"}:
            return True
    return False


def _is_neutral_regularizer_only(cons: tuple[Constraint, ...]) -> bool:
    if not cons:
        return False

    needed = {(d, ">=", _NEUTRAL_LEVEL) for d in _NEUTRAL_DIMS} | {(d, "<=", _NEUTRAL_LEVEL) for d in _NEUTRAL_DIMS}
    got = set()

    for c in cons:
        if c.cutpoint is not None:
            return False
        if c.level is None:
            return False
        if str(c.dim) not in _NEUTRAL_DIMS:
            return False
        if str(c.level) != _NEUTRAL_LEVEL:
            return False
        if c.op not in {">=", "<="}:
            return False

        got.add((str(c.dim), str(c.op), str(c.level)))

        if float(getattr(c, "weight", 0.0) or 0.0) > 0.35:
            return False

    return got == needed


def _is_brightness_only_constraints(nlp_res) -> bool:
    trace = _get(nlp_res, "trace", None) or {}
    cons = trace.get("constraints_final", []) or []
    if not cons:
        return False

    for c in cons:
        axis = c.get("axis") if isinstance(c, dict) else None
        if not isinstance(axis, str):
            return False
        ax = axis.strip().lower()
        if ax not in {"brightness", "lightness"}:
            return False
    return True


# ---------------------------------------------------------------------
# Constraint feasibility (for pool expansion, no clusters)
# ---------------------------------------------------------------------


def _fixed_threshold_from_calibration(cal: dict, dim: str, level: str) -> float:
    th = cal.get("thresholds", {}).get(dim, {})
    if not isinstance(th, dict) or level not in th:
        raise KeyError(f"Missing calibration threshold for dim={dim!r} level={level!r}")
    return float(th[level])


def _constraint_threshold_for_pool(cal: dict, c: Constraint) -> float:
    if c.cutpoint is not None:
        return float(c.cutpoint)
    if c.level is None:
        raise ValueError(f"Constraint has neither cutpoint nor level: {c}")
    return _fixed_threshold_from_calibration(cal, c.dim, str(c.level))


def _satisfy_constraint_series(vals: pd.Series, c: Constraint, thr: float) -> pd.Series:
    x = pd.to_numeric(vals, errors="coerce")
    if c.op == "<=":
        return x <= float(thr)
    return x >= float(thr)


def _print_dim_stats(df: pd.DataFrame, dim: str) -> None:
    if dim not in df.columns or df.empty:
        return
    vals = pd.to_numeric(df[dim], errors="coerce").dropna()
    if vals.empty:
        return
    print(f"  {dim:<12} min={vals.min():.3f}  mean={vals.mean():.3f}  max={vals.max():.3f}")


def _restore_explain_cols(
    base_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    *,
    explain_cols: tuple[str, ...] = _EXPLAIN_COLS,
) -> pd.DataFrame:
    if scored_df is None or scored_df.empty:
        return scored_df
    if base_df is None or base_df.empty:
        return scored_df

    need = [c for c in explain_cols if c in base_df.columns and c not in scored_df.columns]
    if not need:
        return scored_df

    key_candidates = [("product_id", "shade_id"), ("shade_id",), ("product_id",)]
    keys: list[str] | None = None
    for kc in key_candidates:
        if all(k in base_df.columns for k in kc) and all(k in scored_df.columns for k in kc):
            keys = list(kc)
            break

    if keys is None:
        if base_df.index.equals(scored_df.index):
            out = scored_df.copy()
            for c in need:
                out[c] = base_df[c]
            return out
        return scored_df

    out = scored_df.copy()
    feat = base_df[keys + need].drop_duplicates(subset=keys).copy()

    for k in keys:
        out[k] = out[k].astype(str)
        feat[k] = feat[k].astype(str)

    out = out.merge(feat, on=keys, how="left")
    return out


def _standardize_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()

    if "score_total" not in out.columns and "score" in out.columns:
        out = out.rename(columns={"score": "score_total"})

    if "shade" not in out.columns:
        for c in ["shade_name", "name", "variant_name", "item_name", "product_name"]:
            if c in out.columns:
                out["shade"] = out[c].astype(str)
                break

    return out


@lru_cache(maxsize=1)
def _queryspec_param_names() -> set[str]:
    try:
        sig = inspect.signature(QuerySpec)
        return {p.name for p in sig.parameters.values()}
    except Exception:
        return set()


def _make_query_spec(
    *,
    constraints: tuple[Constraint, ...],
    anchor_lab: Optional[Tuple[float, float, float]],
) -> QuerySpec:
    params = _queryspec_param_names()
    kw = {}

    if "constraints" in params:
        kw["constraints"] = tuple(constraints)

    if anchor_lab is not None:
        L, a, b = anchor_lab

        for k in ("anchor_lab", "anchor", "anchor_Lab", "target_lab", "lab"):
            if k in params:
                kw[k] = (float(L), float(a), float(b))
                break

        if ("anchor_L" in params and "anchor_a" in params and "anchor_b" in params):
            kw["anchor_L"] = float(L)
            kw["anchor_a"] = float(a)
            kw["anchor_b"] = float(b)

        if ("lab_L" in params and "lab_a" in params and "lab_b" in params):
            kw["lab_L"] = float(L)
            kw["lab_a"] = float(a)
            kw["lab_b"] = float(b)

    return QuerySpec(**kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Pool distance: angle(a,b) + chroma + lightness (hue lock)
# ---------------------------------------------------------------------


def _hue_angle_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(b, a)


def _wrap_delta_rad(d: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(d), np.cos(d))


def _hue_lock_d2(df: pd.DataFrame, anchor_lab: Tuple[float, float, float]) -> np.ndarray:
    L0, a0, b0 = anchor_lab

    L = pd.to_numeric(df["L_lab"], errors="coerce").to_numpy(float)
    a = pd.to_numeric(df["a_lab"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(df["b_lab"], errors="coerce").to_numpy(float)

    C = np.sqrt(a * a + b * b)
    C0 = float(np.sqrt(float(a0) * float(a0) + float(b0) * float(b0)))

    th = _hue_angle_rad(a, b)
    th0 = float(np.arctan2(float(b0), float(a0)))
    dth = _wrap_delta_rad(th - th0)

    wL = float(_NEIGHBOR_WL)
    wT = float(_NEIGHBOR_WTHETA)
    wC = float(_NEIGHBOR_WC)

    d2 = wL * (L - float(L0)) ** 2 + wT * (dth**2) + wC * (C - C0) ** 2
    return d2


def _select_candidate_pool_distance_only(
    inv: pd.DataFrame,
    *,
    anchor_lab: Optional[Tuple[float, float, float]],
    has_color: bool,
    pool_topn: int,
) -> pd.DataFrame:
    if (not has_color) or (anchor_lab is None):
        return inv

    need = {"L_lab", "a_lab", "b_lab"}
    if not need.issubset(set(inv.columns)):
        return inv

    n = int(min(max(int(pool_topn), 1), len(inv)))
    d2 = _hue_lock_d2(inv, anchor_lab)
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    idx = np.argpartition(d2, kth=min(n - 1, len(d2) - 1))[:n]
    out = inv.iloc[idx].copy()
    out["_d2_anchor"] = d2[idx]
    out = out.sort_values("_d2_anchor", ascending=True).drop(columns=["_d2_anchor"])
    return out


# ---------------------------------------------------------------------
# Hue-window pool + neutral center (plain-color mode)
# ---------------------------------------------------------------------


def _delta_angle_deg(a1: np.ndarray, a0: float) -> np.ndarray:
    d = (a1 - float(a0) + 180.0) % 360.0 - 180.0
    return np.abs(d)


def _lab_hue_angle_deg_from_ab(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(b, a))


def _anchor_hue_angle_deg(anchor_lab: Tuple[float, float, float]) -> float:
    _, a0, b0 = anchor_lab
    return float(np.degrees(np.arctan2(float(b0), float(a0))))


def _pool_by_hue_window(
    inv: pd.DataFrame,
    *,
    anchor_lab: Tuple[float, float, float],
    angle_deg: float,
) -> pd.DataFrame:
    need = {"a_lab", "b_lab"}
    if not need.issubset(inv.columns):
        return inv

    a = pd.to_numeric(inv["a_lab"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(inv["b_lab"], errors="coerce").to_numpy(float)

    th = _lab_hue_angle_deg_from_ab(a, b)
    th0 = _anchor_hue_angle_deg(anchor_lab)

    d = _delta_angle_deg(th, th0)
    m = np.isfinite(d) & (d <= float(angle_deg))
    return inv.loc[m].copy()


def _choose_plain_color_pool(
    inv: pd.DataFrame,
    *,
    anchor_lab: Tuple[float, float, float],
    topk: int,
    debug: bool,
) -> tuple[pd.DataFrame, dict]:
    min_pool = max(50, int(_MIN_FEASIBLE_MULT) * int(topk))

    pool25 = _pool_by_hue_window(inv, anchor_lab=anchor_lab, angle_deg=float(_HUE_WINDOW_DEG_DENSE))
    n_close_25 = int(len(pool25))

    # FIX: "dense" means: enough candidates to be reliable
    dense = n_close_25 >= int(min_pool)
    low_color_coverage = n_close_25 < int(min_pool)

    chosen_angle = None
    chosen_pool = None

    if dense:
        chosen_angle = float(_HUE_WINDOW_DEG_DENSE)
        chosen_pool = pool25
    else:
        # Expand window to get a workable pool, but keep low_color_coverage signal
        for ang in _HUE_WINDOW_DEG_STEPS:
            p = _pool_by_hue_window(inv, anchor_lab=anchor_lab, angle_deg=float(ang))
            if len(p) >= min_pool:
                chosen_angle = float(ang)
                chosen_pool = p
                break
        if chosen_pool is None:
            chosen_angle = float(_HUE_WINDOW_DEG_STEPS[-1])
            chosen_pool = _pool_by_hue_window(inv, anchor_lab=anchor_lab, angle_deg=chosen_angle)

    meta = {
        "min_pool": int(min_pool),
        "n_close_25": int(n_close_25),
        "dense": bool(dense),
        "angle_deg": float(chosen_angle),
        "pool_n": int(len(chosen_pool)),
        "low_color_coverage": bool(low_color_coverage),
    }

    if debug:
        print("\n[Plain color pool]")
        for k, v in meta.items():
            print(f"  {k}={v}")

    return chosen_pool, meta


def _robust_center_lab(df_pool: pd.DataFrame) -> Optional[Tuple[float, float, float]]:
    need = {"L_lab", "a_lab", "b_lab"}
    if not need.issubset(df_pool.columns):
        return None

    L = pd.to_numeric(df_pool["L_lab"], errors="coerce").to_numpy(float)
    a = pd.to_numeric(df_pool["a_lab"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(df_pool["b_lab"], errors="coerce").to_numpy(float)

    ok = np.isfinite(L) & np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return None

    return (float(np.median(L[ok])), float(np.median(a[ok])), float(np.median(b[ok])))


def _deltaE76_to_center(df: pd.DataFrame, center_lab: Tuple[float, float, float]) -> pd.Series:
    need = {"L_lab", "a_lab", "b_lab"}
    if not need.issubset(df.columns):
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    L = pd.to_numeric(df["L_lab"], errors="coerce").to_numpy(float)
    a = pd.to_numeric(df["a_lab"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(df["b_lab"], errors="coerce").to_numpy(float)

    L0, a0, b0 = center_lab
    d = np.sqrt((L - float(L0)) ** 2 + (a - float(a0)) ** 2 + (b - float(b0)) ** 2)
    d = np.where(np.isfinite(d), d, np.inf)
    return pd.Series(d, index=df.index, dtype=float)


def _count_feasible_rows(df: pd.DataFrame, *, constraints: tuple[Constraint, ...], calibration: dict) -> int:
    if df.empty or (not constraints):
        return int(len(df))
    needed_dims = {c.dim for c in constraints}
    if any(d not in df.columns for d in needed_dims):
        return 0

    m = pd.Series(True, index=df.index)
    for c in constraints:
        thr = _constraint_threshold_for_pool(calibration, c)
        m &= _satisfy_constraint_series(df[c.dim], c, thr).fillna(False)
    return int(m.sum())


def recommend_from_text(
    text: str,
    *,
    assets: AssetBundle | None = None,
    topk: int = 20,
    candidate_pool_topn: int = _DEFAULT_POOL_TOPN,
    lambda_constraints: float = 2.0,
    lambda_preference: float = 0.0,
    constraints_blob_override: str | None = None,
    thresholds_override: dict[Axis, AxisThreshold] | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    if assets is None:
        assets = load_default_assets()

    nlp_res = interpret_nlp(text, debug=debug)
    lambda_preference = float(lambda_preference)

    has_color = _has_color_like_mention(nlp_res)
    brightness_only = bool(has_color) and _is_brightness_only_constraints(nlp_res)

    seed_hex = _seed_hex_from_nlp(nlp_res)
    anchor_lab = _anchor_lab_from_nlp(nlp_res)

    ths: dict[Axis, AxisThreshold] | None
    constraints_blob: str

    override_used = constraints_blob_override is not None

    if override_used:
        constraints_blob = str(constraints_blob_override).strip()
        ths = None
    else:
        if thresholds_override is not None:
            ths = dict(thresholds_override)
        else:
            resolved = resolve_preference(nlp_res)
            intents_by_axis = project_axes(resolved)
            decisions = merge_axis_intents(intents_by_axis)
            ths = thresholds_from_decisions(decisions)

        constraints_blob = build_constraints_blob_from_thresholds(ths, calibration=assets.calibration)

    cons: tuple[Constraint, ...] = ()
    injected_neutral = False

    if constraints_blob:
        tokens = [t.strip() for t in str(constraints_blob).split(";") if t.strip()]
        cons = tuple(_parse_constraint_token(t) for t in tokens)

    # Inject only when NO override was provided
    if (not override_used) and (not cons) and has_color and (not _has_effective_thresholds(ths)):
        neutral_tokens: list[str] = []
        for d in _NEUTRAL_DIMS:
            neutral_tokens.append(f"{d}>={_NEUTRAL_LEVEL}:{_NEUTRAL_W}")
            neutral_tokens.append(f"{d}<={_NEUTRAL_LEVEL}:{_NEUTRAL_W}")
        constraints_blob = ";".join(neutral_tokens)
        cons = tuple(_parse_constraint_token(t) for t in neutral_tokens)
        injected_neutral = True

    is_neutral_only = _is_neutral_regularizer_only(cons)

    # Plain-color mode definition (SINGLE source of truth)
    plain_color_mode = bool(has_color) and (not _has_effective_thresholds(ths)) and ((not cons) or is_neutral_only)

    inv = assets.inventory.copy()

    # STEP A: pool selection
    if plain_color_mode and has_color and (anchor_lab is not None):
        df_pool, _pool_meta = _choose_plain_color_pool(inv, anchor_lab=anchor_lab, topk=int(topk), debug=bool(debug))
    else:
        pool_topn = int(min(max(int(candidate_pool_topn), 1), len(inv)))
        df_pool = _select_candidate_pool_distance_only(
            inv,
            anchor_lab=anchor_lab,
            has_color=bool(has_color),
            pool_topn=pool_topn,
        )

        if cons and (not is_neutral_only) and (not brightness_only) and has_color and (anchor_lab is not None):
            min_candidates = max(50, int(_MIN_FEASIBLE_MULT) * int(topk))
            feasible_n = _count_feasible_rows(df_pool, constraints=cons, calibration=assets.calibration)

            if debug:
                print("\n[Pool feasibility]")
                print(f"  pool_topn={len(df_pool)}  feasible_n={feasible_n}  target_min={min_candidates}")

            cur_topn = int(len(df_pool))
            while feasible_n < min_candidates and cur_topn < min(int(_MAX_POOL_TOPN), len(inv)):
                cur_topn = min(cur_topn + int(_POOL_EXPAND_STEP), int(_MAX_POOL_TOPN), len(inv))
                df_pool = _select_candidate_pool_distance_only(
                    inv,
                    anchor_lab=anchor_lab,
                    has_color=True,
                    pool_topn=cur_topn,
                )
                feasible_n = _count_feasible_rows(df_pool, constraints=cons, calibration=assets.calibration)

                if debug:
                    print(f"  expanded_pool_topn={len(df_pool)}  feasible_n={feasible_n}")

    if debug:
        print("\n[NLP constraints_final]")
        for c in (_get(nlp_res, "trace", None) or {}).get("constraints_final", []):
            try:
                print(
                    f"  {c['axis']:>10}  {c['direction']:<5}  "
                    f"{c['strength']:<6}  conf={c['confidence']:.2f}  "
                    f"'{c['evidence']}'"
                )
            except Exception:
                print(f"  {c}")

        if ths:
            print("\n[Axis thresholds]")
            for ax, t in ths.items():
                print(f"  {ax.value:<12} low={t.low} high={t.high} weight={t.weight:.2f}")

        print("\n[Constraints blob]")
        print(f"  {constraints_blob!r}")
        print(f"  injected_neutral={injected_neutral}  neutral_only={is_neutral_only}")
        print(f"  has_color={has_color}  brightness_only={brightness_only}")
        print(f"  neighbor_wL={_NEIGHBOR_WL}  neighbor_wTheta={_NEIGHBOR_WTHETA}  neighbor_wC={_NEIGHBOR_WC}")

        print("\n[Seed hex]")
        print(f"  seed_hex={seed_hex}")

        print("\n[Anchor]")
        print(f"  target_lab={anchor_lab}")

        print("\n[Candidate pool]")
        print(f"  pool_n={len(df_pool)}  inv_n={len(inv)}")

    lambda_constraints_eff = float(lambda_constraints)
    if injected_neutral and is_neutral_only:
        lambda_constraints_eff = min(lambda_constraints_eff, float(_NEUTRAL_LAMBDA_MAX))

    # score_shades contract: if preference_weights is None => lambda_preference MUST be 0
    preference_weights = getattr(assets, "preference_weights", None)
    lambda_preference_eff = float(lambda_preference)

    if preference_weights is None:
        lambda_preference_eff = 0.0
    elif plain_color_mode:
        lambda_preference_eff = max(lambda_preference_eff, float(_PLAIN_COLOR_LAMBDA_PREFERENCE_MIN))

    if debug:
        print("\n[Scoring weights]")
        print(f"  lambda_constraints={float(lambda_constraints):.3f}  lambda_constraints_eff={lambda_constraints_eff:.3f}")
        print(f"  lambda_preference={float(lambda_preference):.3f}  lambda_preference_eff={lambda_preference_eff:.3f}")
        print(f"  preference_weights={'present' if preference_weights is not None else 'None'}")

    query = _make_query_spec(
        constraints=cons,
        anchor_lab=anchor_lab,
    )

    scored = score_shades(
        df_pool,
        None,
        query,
        lambda_constraints=lambda_constraints_eff,
        lambda_preference=float(lambda_preference_eff),
        calibration=assets.calibration,
        preference_weights=preference_weights,
    )

    scored = _restore_explain_cols(df_pool, scored, explain_cols=tuple(_EXPLAIN_COLS))
    scored = _standardize_output_columns(scored)

    # Plain-color: hard guard against globally over-saturated shades (esp. reds)
    if plain_color_mode and (scored is not None) and (not scored.empty) and ("score_total" in scored.columns):
        hi = _global_chroma_hi(inv, _PLAIN_CHROMA_EXTREME_Q)
        if hi is not None:
            scored = _ensure_C_lab(scored)
            c = pd.to_numeric(scored["C_lab"], errors="coerce").fillna(0.0)
            over = np.maximum(0.0, c - float(hi))
            scored["_chroma_over_hi"] = over
            scored["score_total"] = pd.to_numeric(scored["score_total"], errors="coerce").fillna(0.0) - float(
                _PLAIN_CHROMA_EXTREME_LAMBDA
            ) * over

    # STEP B: neutral target = robust center of the (clean) pool
    if plain_color_mode and (scored is not None) and (not scored.empty) and ("score_total" in scored.columns):
        center_lab = _robust_center_lab(df_pool)
        if center_lab is not None:
            dE = _deltaE76_to_center(scored, center_lab)
            scored["_dE_center"] = dE
            scored["score_total"] = pd.to_numeric(scored["score_total"], errors="coerce").fillna(0.0) - float(
                _CENTER_DE_LAMBDA
            ) * dE

    apply_canonical = plain_color_mode
    if apply_canonical and (scored is not None) and (not scored.empty) and ("score_total" in scored.columns):
        pen = _canonical_penalty(scored)
        scored["_canonical_penalty"] = pen
        scored["_score_canonical"] = -pd.to_numeric(pen, errors="coerce").fillna(0.0)
        scored["score_total"] = pd.to_numeric(scored["score_total"], errors="coerce").fillna(0.0) + (
            float(_CANONICAL_LAMBDA) * scored["_score_canonical"]
        )

    # Typical-color rerank must ONLY run for plain/neutral color queries.
    apply_typical = bool(seed_hex) and bool(has_color) and bool(plain_color_mode)

    if apply_typical:
        scored = apply_typical_median_near_seed(
            scored,
            seed_hex=seed_hex,
            chip_hex_col="chip_hex",
            score_col="score_total",
            neighbor_m=int(_TYPICAL_NEIGHBOR_M),
            min_n=int(_TYPICAL_MIN_N),
            weight=float(_TYPICAL_WEIGHT),
        )

    sort_col = "score_total" if "score_total" in scored.columns else ("score" if "score" in scored.columns else None)
    if sort_col is None:
        top = scored.head(int(topk)).copy()
    else:
        top = scored.sort_values(sort_col, ascending=False).head(int(topk)).copy()

    if debug:
        print("\n[Top-K dim distribution]")
        for dim in ["light_hsl", "sat_hsl", "warmth", "depth", "colorfulness", "C_lab", "L_lab"]:
            _print_dim_stats(top, dim)

    if plain_color_mode and isinstance(locals().get("_pool_meta", None), dict):
        pm = _pool_meta
        for k, v in {
            "_pool_dense": pm.get("dense"),
            "_pool_n_close_25": pm.get("n_close_25"),
            "_pool_angle_deg": pm.get("angle_deg"),
            "_pool_n": pm.get("pool_n"),
            "_low_color_coverage": pm.get("low_color_coverage"),
        }.items():
            top[k] = v

    return top


__all__ = [
    "recommend_from_text",
]
