# src/Shopping_assistant/color/scoring.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "Constraint",
    "QuerySpec",
    "constraints_from_nlp",
    "score_shades",
    "score_inventory",
    "load_scoring_calibration",
    "load_preference_weights",
]

# ---------------------------------------------------------------------
# Script/CLI defaults (NOT for library use)
# ---------------------------------------------------------------------


def _project_root_for_scripts_only() -> Path:
    # .../src/Shopping_assistant/color/scoring.py -> parents[3] = project root
    return Path(__file__).resolve().parents[3]


def _default_enriched_for_scripts_only() -> Path:
    return (
        _project_root_for_scripts_only()
        / "data"
        / "enriched_data"
        / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv"
    )


def _default_outdir_for_scripts_only() -> Path:
    return _project_root_for_scripts_only() / "data" / "scores"


def _default_calibration_for_scripts_only() -> Path:
    return _project_root_for_scripts_only() / "data" / "models" / "color_scoring_calibration.json"


def _default_preference_weights_for_scripts_only() -> Path:
    return _project_root_for_scripts_only() / "data" / "models" / "color_preference_weights.json"


def _rgb_cols_to_hex(df: pd.DataFrame) -> pd.Series:
    """
    Convert columns r,g,b (0..255) into #RRGGBB.
    """
    r = df["r"].astype("Int64")
    g = df["g"].astype("Int64")
    b = df["b"].astype("Int64")
    ok = r.notna() & g.notna() & b.notna()
    hexs = pd.Series(pd.NA, index=df.index, dtype="string")
    hexs.loc[ok] = (
        "#"
        + r.loc[ok].astype(int).map(lambda x: f"{x:02X}")
        + g.loc[ok].astype(int).map(lambda x: f"{x:02X}")
        + b.loc[ok].astype(int).map(lambda x: f"{x:02X}")
    )
    return hexs


# ---------------------------------------------------------------------
# Pure loaders (explicit paths only)
# ---------------------------------------------------------------------


def load_scoring_calibration(path: str | Path) -> dict:
    cal_path = Path(path)
    if not cal_path.exists():
        raise FileNotFoundError(f"Missing scoring calibration file: {cal_path}")
    payload = json.loads(cal_path.read_text(encoding="utf-8"))

    # minimal validation
    for k in ("deltaE_ref", "thresholds", "scale_iqr", "scale_std"):
        if k not in payload:
            raise KeyError(f"Invalid calibration file (missing '{k}'): {cal_path}")
    if not np.isfinite(float(payload["deltaE_ref"])) or float(payload["deltaE_ref"]) <= 0:
        raise ValueError(f"Invalid calibration deltaE_ref: {payload.get('deltaE_ref')}")
    return payload


def load_preference_weights(path: str | Path) -> Optional[dict[str, float]]:
    p = Path(path)
    if not p.exists():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    w = payload.get("weights")
    return w if isinstance(w, dict) else None


# ---------------------------------------------------------------------
# Query / Constraint model (generic)
# ---------------------------------------------------------------------

_ALLOWED_DIMS = {
    "L_lab",
    "a_lab",
    "b_lab",
    "C_lab",
    "H_lab_deg",
    "depth",
    "warmth",
    "sat_eff",
    "sat_hsl",
    "light_hsl",
    "Y_rel",
    "colorfulness",
}

_ALLOWED_OPS = {"<=", ">="}


@dataclass(frozen=True)
class Constraint:
    dim: str
    op: str
    level: str | None = None  # low|medium|high|very_high
    cutpoint: float | None = None  # numeric threshold (preferred when provided)
    weight: float = 1.0


@dataclass(frozen=True)
class QuerySpec:
    constraints: Tuple[Constraint, ...] = ()
    # anchor used when prototypes are not provided (or to override prototype center)
    anchor_lab: Optional[Tuple[float, float, float]] = None


# --- NLP -> scoring constraint adapter ---------------------------------


def _nlp_axis_to_dim(axis: str) -> str | None:
    # Map NLP axes to calibrated scoring dims
    if axis == "brightness":
        return "light_hsl"
    if axis == "saturation":
        return "sat_hsl"
    if axis == "vibrancy":
        return "sat_eff"
    if axis == "depth":
        return "depth"
    if axis == "clarity":
        return "colorfulness"
    return None


def _nlp_strength_to_level(direction: str, strength: str) -> str:
    """
    direction: 'raise'|'lower'
    strength: 'weak'|'med'|'strong'
    Output: 'low'|'medium'|'high'|'very_high'

    FIX:
      For "lower", stronger intent must push the threshold LOWER (level 'low').
    """
    strength = str(strength)
    direction = str(direction)

    if direction == "lower":
        # "lower X" => <= threshold. strong => low, med => medium, weak => high.
        return {"strong": "low", "med": "medium", "weak": "high"}.get(strength, "medium")

    # "raise X" => >= threshold. strong => high, med => medium, weak => low.
    return {"strong": "high", "med": "medium", "weak": "low"}.get(strength, "medium")


def _nlp_strength_to_weight(strength: str, *, soft_weight: float) -> float:
    base = {"weak": 0.5, "med": 0.8, "strong": 1.0}.get(strength, 0.8)
    if str(strength) != "strong":
        return base * float(soft_weight)
    return base


def constraints_from_nlp(
    nlp_constraints: Sequence[object],
    *,
    soft_weight: float = 0.5,
) -> Tuple["Constraint", ...]:
    """
    Does:
        Convert NLP constraints (axis/direction/strength/meta) into scoring constraints (dim/op/level/weight).
        If meta['axis_query'] exists and is a valid scoring dim, it overrides the default axis->dim mapping.
    """
    out: List[Constraint] = []
    for c in nlp_constraints:
        axis = getattr(getattr(c, "axis", None), "value", None)
        direction = getattr(getattr(c, "direction", None), "value", None)
        strength = getattr(getattr(c, "strength", None), "value", None)
        meta = getattr(c, "meta", None) or {}

        if not axis or not direction or not strength:
            continue

        # Override dim via meta["axis_query"] when present
        dim_override = None
        if isinstance(meta, dict):
            dim_override = meta.get("axis_query")

        if isinstance(dim_override, str) and dim_override in _ALLOWED_DIMS:
            dim = dim_override
        else:
            dim = _nlp_axis_to_dim(str(axis))
            if dim is None:
                continue

        op = "<=" if str(direction) == "lower" else ">="
        level = _nlp_strength_to_level(str(direction), str(strength))
        weight = _nlp_strength_to_weight(str(strength), soft_weight=float(soft_weight))

        out.append(Constraint(dim=str(dim), op=str(op), level=level, cutpoint=None, weight=weight))

    return tuple(out)


# ---------------------------------------------------------------------
# Public API helpers
# ---------------------------------------------------------------------


def _split_constraints_blob(constraints: str) -> List[str]:
    """Split scenario constraint blob into tokens (semicolon separated)."""
    if not constraints:
        return []
    raw = str(constraints).replace("\n", ";")
    return [c.strip() for c in raw.split(";") if c.strip()]


def _parse_mix_and_constraints(constraints: str) -> Tuple[Optional[dict], List[str]]:
    """
    Parse a combined constraints blob that may include a MIX block.

    NOTE (no-clusters):
      MIX is still supported as an "anchor mixer" when prototypes exist.
      If prototypes are absent, MIX requires anchor_lab_override at call sites (not provided here).
    """
    tokens = _split_constraints_blob(constraints)
    if not tokens:
        return None, []

    mix: Optional[dict] = None
    out: List[str] = []

    in_mix = False
    for t in tokens:
        up = t.upper()

        if up.startswith("MIX:"):
            in_mix = True
            tail = t.split(":", 1)[1].strip()
            if not tail:
                continue

            if ("<=" in tail) or (">=" in tail):
                out.append(tail)
                continue

            if mix is None:
                mix = {}
            if "=" not in tail:
                raise ValueError(f"Invalid MIX token '{t}'.")
            k, v = tail.split("=", 1)
            mix[k.strip().lower()] = v.strip()
            continue

        if in_mix:
            if ("<=" in t) or (">=" in t):
                in_mix = False
                out.append(t)
                continue

            if mix is None:
                mix = {}
            if "=" not in t:
                raise ValueError(f"Invalid MIX token '{t}'.")
            k, v = t.split("=", 1)
            mix[k.strip().lower()] = v.strip()
            continue

        out.append(t)

    if mix is None:
        return None, out

    try:
        primary = int(mix["primary"])
        secondary = int(mix["secondary"])
        alpha = float(mix["alpha"])
    except Exception as e:
        raise ValueError(f"Invalid MIX block in constraints: {constraints!r}") from e

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Invalid MIX alpha={alpha} (expected in [0,1]).")

    return {"primary": primary, "secondary": secondary, "alpha": alpha}, out


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------


def _require_cols(df: pd.DataFrame, cols: Sequence[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def _check_constraint(c: Constraint) -> None:
    if c.dim not in _ALLOWED_DIMS:
        raise ValueError(f"Unknown dim '{c.dim}'.")
    if c.op not in _ALLOWED_OPS:
        raise ValueError(f"Unknown op '{c.op}'.")
    if c.cutpoint is None:
        if c.level not in {"low", "medium", "high", "very_high"}:
            raise ValueError(f"Unknown level '{c.level}'.")
    else:
        if not np.isfinite(c.cutpoint):
            raise ValueError(f"Invalid cutpoint '{c.cutpoint}'.")
    if not np.isfinite(c.weight) or c.weight < 0:
        raise ValueError(f"Invalid weight '{c.weight}'.")


# ---------------------------------------------------------------------
# Color distance (ΔE CIE76) + normalization
# ---------------------------------------------------------------------


def _delta_e_to_proto(work: pd.DataFrame, proto_row: pd.Series) -> np.ndarray:
    dL = work["L_lab"].to_numpy(float) - float(proto_row["L_lab"])
    da = work["a_lab"].to_numpy(float) - float(proto_row["a_lab"])
    db = work["b_lab"].to_numpy(float) - float(proto_row["b_lab"])
    return np.sqrt(dL * dL + da * da + db * db)


def _delta_e_norm(deltaE: np.ndarray, cal: dict) -> np.ndarray:
    ref = float(cal["deltaE_ref"])
    return deltaE / ref


# ---------------------------------------------------------------------
# Constraints (absolute vs relative thresholds + normalized penalties)
# ---------------------------------------------------------------------


def _constraint_threshold_fixed(cal: dict, dim: str, level: str) -> float:
    try:
        return float(cal["thresholds"][dim][level])
    except Exception as e:
        raise KeyError(f"Missing fixed threshold for dim={dim!r} level={level!r} in calibration.") from e


def _dim_cfg(cal: dict | None, dim: str) -> dict:
    if not cal:
        return {}
    dims = cal.get("dims") or cal.get("dimensions") or {}
    if not isinstance(dims, dict):
        return {}
    cfg = dims.get(dim) or {}
    return cfg if isinstance(cfg, dict) else {}


def _dim_level_mode(cal: dict | None, dim: str) -> str:
    cfg = _dim_cfg(cal, dim)
    mode = cfg.get("level_mode") or cfg.get("mode") or "absolute"
    return str(mode).strip().lower()


def _dim_relative_quantiles(cal: dict | None, dim: str) -> dict[str, float]:
    cfg = _dim_cfg(cal, dim)
    q = cfg.get("relative_quantiles")
    if isinstance(q, dict) and q:
        out: dict[str, float] = {}
        for k, v in q.items():
            try:
                out[str(k).lower()] = float(v)
            except Exception:
                continue
        if out:
            return out
    return {"low": 0.70, "medium": 0.80, "high": 0.90, "very_high": 0.95}


def _calib_bounds_for_dim(calibration: dict, *, dim: str) -> tuple[float | None, float | None]:
    ths = calibration.get("thresholds", {}).get(dim)
    if not isinstance(ths, dict) or not ths:
        return None, None
    vals = []
    for _, v in ths.items():
        try:
            vv = float(v)
            if np.isfinite(vv):
                vals.append(vv)
        except Exception:
            continue
    if not vals:
        return None, None
    return float(min(vals)), float(max(vals))


def _constraint_threshold_relative(
    work: pd.DataFrame,
    *,
    cal: dict,
    dim: str,
    level: str,
    op: str,
) -> float:
    if dim not in work.columns or work.empty:
        raise KeyError(f"Cannot compute relative threshold: missing dim '{dim}' or empty pool.")
    vals = pd.to_numeric(work[dim], errors="coerce").dropna()
    if vals.empty:
        raise ValueError(f"Cannot compute relative threshold: no finite values for dim '{dim}'.")

    cfg = _dim_cfg(cal, dim)
    min_pool_n = int(cfg.get("min_pool_n", 0) or 0)
    if min_pool_n and len(vals) < min_pool_n:
        return _constraint_threshold_fixed(cal, dim, level)

    qmap = _dim_relative_quantiles(cal, dim)
    q = float(qmap.get(str(level).lower(), 0.80))

    if str(op).strip() == "<=":
        q = 1.0 - q

    q = min(0.999, max(0.001, q))
    thr = float(vals.quantile(q))

    if bool(cfg.get("clamp_to_fixed_bounds", False)):
        lo, hi = _calib_bounds_for_dim(cal, dim=dim)
        if lo is not None and np.isfinite(lo):
            thr = max(thr, float(lo))
        if hi is not None and np.isfinite(hi):
            thr = min(thr, float(hi))

    return float(thr)


def _constraint_threshold_dynamic(work: pd.DataFrame, cal: dict, c: "Constraint") -> float:
    if c.cutpoint is not None:
        return float(c.cutpoint)
    if c.level is None:
        raise ValueError(f"Constraint has neither cutpoint nor level: {c}")

    mode = _dim_level_mode(cal, c.dim)
    if mode == "relative":
        return _constraint_threshold_relative(work, cal=cal, dim=c.dim, level=str(c.level), op=str(c.op))

    return _constraint_threshold_fixed(cal, c.dim, str(c.level))


def _constraint_scale(cal: dict, dim: str) -> float:
    iqr = float(cal["scale_iqr"].get(dim, float("nan")))
    if np.isfinite(iqr) and iqr > 1e-12:
        return iqr
    std = float(cal["scale_std"].get(dim, float("nan")))
    if np.isfinite(std) and std > 1e-12:
        return std
    return 1.0


def _constraint_violation(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    """
    Hinge violation (always >= 0):
      - "<=" : violation = max(0, values - threshold)
      - ">=" : violation = max(0, threshold - values)

    penalty-only. Must NOT reward pushing values to extremes.
    """
    if op == "<=":
        return np.maximum(0.0, values - threshold)
    return np.maximum(0.0, threshold - values)


def _constraint_suffix(c: Constraint) -> str:
    if c.cutpoint is not None:
        return f"{float(c.cutpoint):.6g}"
    return str(c.level)


def _apply_constraints(
    work: pd.DataFrame,
    constraints: Tuple[Constraint, ...],
    *,
    calibration: dict,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if not constraints:
        z = np.zeros(len(work), float)
        return z, pd.DataFrame(index=work.index), pd.DataFrame(index=work.index)

    total = np.zeros(len(work), float)
    breakdown: dict[str, np.ndarray] = {}
    extras: dict[str, np.ndarray] = {}

    for i, c in enumerate(constraints, 1):
        _check_constraint(c)
        if c.dim not in work.columns:
            raise KeyError(f"Constraint dim '{c.dim}' missing from work df columns.")

        thr = _constraint_threshold_dynamic(work, calibration, c)
        scale = _constraint_scale(calibration, c.dim)

        vals = pd.to_numeric(work[c.dim], errors="coerce").to_numpy(float)
        safe_vals = np.nan_to_num(vals, nan=float(thr))

        viol = _constraint_violation(safe_vals, c.op, float(thr))
        viol_norm = viol / float(scale)

        pen = float(c.weight) * viol_norm
        total += pen

        breakdown[f"penalty_{i}__{c.dim}{c.op}{_constraint_suffix(c)}"] = pen
        extras[f"c{i}__{c.dim}"] = safe_vals

    return total, pd.DataFrame(breakdown, index=work.index), pd.DataFrame(extras, index=work.index)


def _constraint_threshold(calibration: dict, c: Constraint) -> float:
    if c.cutpoint is not None:
        return float(c.cutpoint)
    return _constraint_threshold_fixed(calibration, c.dim, c.level)  # type: ignore[arg-type]


def _satisfy_mask(work: pd.DataFrame, c: Constraint, thr: float) -> np.ndarray:
    vals = pd.to_numeric(work[c.dim], errors="coerce").to_numpy(float)
    vals = np.nan_to_num(vals, nan=float(thr))
    if c.op == "<=":
        return vals <= float(thr)
    return vals >= float(thr)


def _snap_numeric_cutpoints_to_joint_feasibility(
    work: pd.DataFrame,
    constraints: Tuple[Constraint, ...],
    *,
    calibration: dict,
    min_feasible_frac: float = 0.05,
    max_iter: int = 12,
    clamp_to_calibration: bool = True,
) -> Tuple[Constraint, ...]:
    if not constraints:
        return constraints

    mf = float(min_feasible_frac)
    if not (0.0 < mf < 0.5):
        return constraints

    any_numeric = any(c.cutpoint is not None for c in constraints)
    if not any_numeric:
        return constraints

    thr_map: List[float] = [_constraint_threshold(calibration, c) for c in constraints]

    def mask_all_current(except_idx: int | None = None) -> np.ndarray:
        m = np.ones(len(work), dtype=bool)
        for j, c in enumerate(constraints):
            if except_idx is not None and j == except_idx:
                continue
            if c.dim not in work.columns:
                continue
            m &= _satisfy_mask(work, c, thr_map[j])
        return m

    m0 = mask_all_current()
    if float(m0.mean()) >= mf:
        return constraints

    bounds_cache: dict[str, tuple[float | None, float | None]] = {}

    for _ in range(int(max_iter)):
        m_all = mask_all_current()
        if float(m_all.mean()) >= mf:
            break

        changed = False

        for i, c in enumerate(constraints):
            if c.cutpoint is None:
                continue
            if c.dim not in work.columns:
                continue

            m_other = mask_all_current(except_idx=i)
            if not m_other.any():
                m_other = np.ones(len(work), dtype=bool)

            vals = pd.to_numeric(work.loc[m_other, c.dim], errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue

            cur = float(thr_map[i])

            if c.op == "<=":
                target = float(np.quantile(vals, mf))
                new_thr = max(cur, target)
            elif c.op == ">=":
                target = float(np.quantile(vals, 1.0 - mf))
                new_thr = min(cur, target)
            else:
                continue

            if clamp_to_calibration:
                if c.dim not in bounds_cache:
                    bounds_cache[c.dim] = _calib_bounds_for_dim(calibration, dim=c.dim)
                lo, hi = bounds_cache[c.dim]
                if c.op == "<=" and hi is not None and np.isfinite(hi):
                    new_thr = min(float(new_thr), float(hi))
                if c.op == ">=" and lo is not None and np.isfinite(lo):
                    new_thr = max(float(new_thr), float(lo))

            if np.isfinite(new_thr) and abs(float(new_thr) - cur) > 1e-12:
                thr_map[i] = float(new_thr)
                changed = True

        if not changed:
            break

    out: List[Constraint] = []
    for i, c in enumerate(constraints):
        if c.cutpoint is None:
            out.append(c)
        else:
            out.append(Constraint(dim=c.dim, op=c.op, level=c.level, cutpoint=float(thr_map[i]), weight=c.weight))
    return tuple(out)


# ---------------------------------------------------------------------
# Preference features for anchor-based scoring (no prototypes)
# ---------------------------------------------------------------------


def _lab_to_C_H(Lab: Tuple[float, float, float]) -> tuple[float, float]:
    _, a, b = Lab
    C = float(np.sqrt(float(a) ** 2 + float(b) ** 2))
    H = float(np.degrees(np.arctan2(float(b), float(a))))
    if H < 0.0:
        H += 360.0
    return C, H


def _proto_row_from_anchor(
    *,
    anchor_lab: Tuple[float, float, float],
) -> pd.Series:
    L0, a0, b0 = anchor_lab
    C0, H0 = _lab_to_C_H((float(L0), float(a0), float(b0)))
    return pd.Series(
        {
            "L_lab": float(L0),
            "a_lab": float(a0),
            "b_lab": float(b0),
            "C_lab": float(C0),
            "H_lab_deg": float(H0),
        }
    )


# ---------------------------------------------------------------------
# Scoring (CORE) - no implicit disk access
# ---------------------------------------------------------------------


def score_shades(
    df: pd.DataFrame,
    prototypes_or_query: pd.DataFrame | "QuerySpec" | None,
    query: "QuerySpec" | None = None,
    *,
    lambda_constraints: float = 1.0,
    lambda_preference: float = 1.0,
    proto_row_override: Optional[pd.Series] = None,
    calibration: dict | None = None,
    preference_weights: Optional[dict[str, float]] = None,
    joint_min_feasible_frac: float = 0.05,
    joint_clamp_to_calibration: bool = True,
) -> pd.DataFrame:
    # --- API COMPAT ---
    # Accept:
    #   score_shades(df, query, ...)
    #   score_shades(df, prototypes, query, ...)
    if isinstance(prototypes_or_query, QuerySpec) and query is None:
        prototypes = None
        query = prototypes_or_query
    else:
        prototypes = prototypes_or_query
        if query is None or not isinstance(query, QuerySpec):
            raise TypeError("score_shades expected (df, query, ...) or (df, prototypes, query, ...).")

    if calibration is None:
        raise ValueError(
            "Missing 'calibration' dict. Pass it explicitly (loaded via io/assets or load_scoring_calibration(path))."
        )

    # Accept NLP Constraint objects transparently (axis/direction/strength/meta)
    constraints_in = query.constraints or ()
    if constraints_in and (not hasattr(constraints_in[0], "dim")) and hasattr(constraints_in[0], "axis"):
        constraints_in = constraints_from_nlp(constraints_in)
        query = QuerySpec(constraints=tuple(constraints_in), anchor_lab=query.anchor_lab)

    _require_cols(
        df,
        ("product_id", "shade_id", "brand_name", "product_name", "shade_name", "url", "L_lab", "a_lab", "b_lab"),
        name="df",
    )

    work = df.dropna(subset=["L_lab", "a_lab", "b_lab"]).copy()
    if work.empty:
        raise ValueError("No valid Lab rows to score.")

    # --- prototype/anchor center selection ---
    if proto_row_override is not None:
        proto_row = proto_row_override
    else:
        if prototypes is None:
            if query.anchor_lab is not None:
                proto_row = _proto_row_from_anchor(anchor_lab=query.anchor_lab)
            else:
                # deterministic fallback: mean Lab of pool
                L0 = float(work["L_lab"].astype(float).mean())
                a0 = float(work["a_lab"].astype(float).mean())
                b0 = float(work["b_lab"].astype(float).mean())
                proto_row = _proto_row_from_anchor(anchor_lab=(L0, a0, b0))
        else:
            # cluster-based proto selection is still possible for scripts, but not required by library
            _require_cols(prototypes, ("cluster_id", "L_lab", "a_lab", "b_lab"), name="prototypes")
            # caller must pass the right row through proto_row_override OR provide a 'cluster_id' in prototypes
            # We choose the FIRST row as a safe default if the caller does not override.
            proto_row = prototypes.iloc[0]

    deltaE = _delta_e_to_proto(work, proto_row)
    deltaE_n = _delta_e_norm(deltaE, calibration)

    constraints_eff = _snap_numeric_cutpoints_to_joint_feasibility(
        work,
        query.constraints,
        calibration=calibration,
        min_feasible_frac=float(joint_min_feasible_frac),
        max_iter=12,
        clamp_to_calibration=bool(joint_clamp_to_calibration),
    )

    total_penalty_n, penalty_df, extras_df = _apply_constraints(work, constraints_eff, calibration=calibration)

    if float(lambda_preference) != 0.0:
        if preference_weights is None:
            raise ValueError(
                "lambda_preference != 0 but 'preference_weights' is None. "
                "Pass weights explicitly (load via io/assets or load_preference_weights(path))."
            )

        _require_cols(work, ("C_lab", "H_lab_deg"), name="preference features")
        if "C_lab" not in proto_row.index or "H_lab_deg" not in proto_row.index:
            raise KeyError("proto_row missing required preference columns: ['C_lab', 'H_lab_deg']")

        std_L = _constraint_scale(calibration, "L_lab")
        std_C = _constraint_scale(calibration, "C_lab")

        dL = (work["L_lab"].to_numpy(float) - float(proto_row["L_lab"])) / std_L
        dC = (work["C_lab"].to_numpy(float) - float(proto_row["C_lab"])) / std_C
        dH = (work["H_lab_deg"].to_numpy(float) - float(proto_row["H_lab_deg"]) + 180.0) % 360.0 - 180.0
        dH = dH / 180.0

        wL = abs(float(preference_weights["w_L"]))
        wC = abs(float(preference_weights["w_C"]))
        wH = abs(float(preference_weights["w_H"]))

        pref_dist = (wL * np.abs(dL)) + (wC * np.abs(dC)) + (wH * np.abs(dH))
        preference_score = -pref_dist
    else:
        preference_score = 0.0

    # IMPORTANT: constraints are penalties (>=0), never rewards
    score = -deltaE_n + float(lambda_preference) * preference_score - float(lambda_constraints) * total_penalty_n

    chip_hex_out = None
    if "chip_hex" in work.columns:
        chip_hex_out = work["chip_hex"].astype("string")
    elif {"r", "g", "b"}.issubset(work.columns):
        chip_hex_out = _rgb_cols_to_hex(work)

    out = pd.DataFrame(
        {
            "product_id": work["product_id"].astype(str),
            "shade_id": work["shade_id"].astype(str),
            "brand_name": work["brand_name"].astype(str),
            "product_name": work["product_name"].astype(str),
            "shade_name": work["shade_name"].astype(str),
            "url": work["url"].astype(str),
            "chip_hex": chip_hex_out if chip_hex_out is not None else pd.NA,
            "r": work["r"].astype("Int64") if "r" in work.columns else pd.NA,
            "g": work["g"].astype("Int64") if "g" in work.columns else pd.NA,
            "b": work["b"].astype("Int64") if "b" in work.columns else pd.NA,
            # cluster_id removed from library scoring path; keep if present in df for UX only
            "cluster_id": work["cluster_id"].astype("Int64") if "cluster_id" in work.columns else pd.NA,
            "deltaE": deltaE,
            "deltaE_norm": deltaE_n,
            "constraint_penalty_norm": total_penalty_n,
            "preference_score": preference_score,
            "score": score,
        },
        index=work.index,
    )

    out = pd.concat([out, extras_df, penalty_df], axis=1)
    out = out.sort_values(["score", "product_id", "shade_id"], ascending=[False, True, True], kind="mergesort")

    # keep your original dedupe policy
    out = out.drop_duplicates(subset=["url", "shade_name"], keep="first")

    out = out.reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


# ---------------------------------------------------------------------
# Script-friendly wrapper (explicit paths; defaults only for scripts)
# ---------------------------------------------------------------------


# Single canonical regex (avoid duplicate definitions / divergence)
_CONSTRAINT_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*(<=|>=)\s*((?:low|medium|high|very_high)|(?:[0-9]*\.?[0-9]+))\s*:\s*([0-9]*\.?[0-9]+)\s*$"
)


def _parse_constraint(s: str) -> Constraint:
    raw = s.strip()
    m = _CONSTRAINT_RE.match(raw)
    if not m:
        raise ValueError(f"Invalid constraint '{raw}'.")
    dim, op, lvl_or_num, w_str = m.groups()

    level = None
    cutpoint = None
    if lvl_or_num in {"low", "medium", "high", "very_high"}:
        level = lvl_or_num
    else:
        cutpoint = float(lvl_or_num)

    c = Constraint(dim=str(dim), op=str(op), level=level, cutpoint=cutpoint, weight=float(w_str))
    _check_constraint(c)
    return c


def score_inventory(
    *,
    inventory: pd.DataFrame,
    constraints: str = "",
    lambda_constraints: float = 1.0,
    lambda_preference: float = 0.0,
    anchor_lab: Optional[Tuple[float, float, float]] = None,
    calibration_path: Optional[str | Path] = None,
    preference_weights_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Script wrapper (no clusters):
      - Scores inventory against anchor_lab (if provided) else pool mean.
      - No prototypes, no assignments, no cluster_id required.
    """
    cal_path = Path(calibration_path) if calibration_path is not None else _default_calibration_for_scripts_only()
    cal = load_scoring_calibration(cal_path)

    mix, constraint_tokens = _parse_mix_and_constraints(constraints)
    if mix is not None:
        raise ValueError("MIX is not supported in no-clusters score_inventory(). Use anchor_lab explicitly.")

    parsed: List[Constraint] = [_parse_constraint(s) for s in constraint_tokens]
    query = QuerySpec(constraints=tuple(parsed), anchor_lab=anchor_lab)

    pref_weights = None
    if float(lambda_preference) != 0.0:
        pw_path = (
            Path(preference_weights_path)
            if preference_weights_path is not None
            else _default_preference_weights_for_scripts_only()
        )
        pref_weights = load_preference_weights(pw_path)

    return score_shades(
        inventory,
        None,
        query,
        lambda_constraints=float(lambda_constraints),
        lambda_preference=float(lambda_preference),
        calibration=cal,
        preference_weights=pref_weights,
    )


# ---------------------------------------------------------------------
# CLI (no clusters)
# ---------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--constraint", action="append", default=[])
    p.add_argument("--lambda-constraints", type=float, default=1.0)
    p.add_argument("--lambda-preference", type=float, default=0.0)  # default off
    p.add_argument("--infile", type=str, default=str(_default_enriched_for_scripts_only()))
    p.add_argument("--calibration", type=str, default=str(_default_calibration_for_scripts_only()))
    p.add_argument("--preference-weights", type=str, default=str(_default_preference_weights_for_scripts_only()))
    p.add_argument("--outdir", type=str, default=str(_default_outdir_for_scripts_only()))

    # Anchor options
    p.add_argument("--anchor-lab", type=str, default="", help='Anchor Lab "L,a,b"')
    p.add_argument("--anchor-hex", type=str, default="", help="Anchor hex #RRGGBB (requires r,g,b not used here)")
    return p


def _parse_anchor_lab(s: str) -> Optional[Tuple[float, float, float]]:
    if not s:
        return None
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Invalid --anchor-lab {s!r} (expected 'L,a,b').")
    return float(parts[0]), float(parts[1]), float(parts[2])


def main() -> None:
    args = _build_argparser().parse_args()

    df = pd.read_csv(Path(args.infile))

    constraints = ";".join(args.constraint or [])
    anchor_lab = _parse_anchor_lab(args.anchor_lab)

    scored = score_inventory(
        inventory=df,
        constraints=constraints,
        lambda_constraints=float(args.lambda_constraints),
        lambda_preference=float(args.lambda_preference),
        anchor_lab=anchor_lab,
        calibration_path=Path(args.calibration),
        preference_weights_path=Path(args.preference_weights),
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "scores_anchor.csv"
    scored.to_csv(outpath, index=False)
    print(f"[OK] Wrote {outpath}")


if __name__ == "__main__":
    main()
