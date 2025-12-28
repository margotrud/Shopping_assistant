# src/Shopping_assistant/color/scoring.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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


def _default_prototypes_for_scripts_only() -> Path:
    """
    Prefer fused Lab prototypes if present; otherwise fallback to kmeans prototypes.
    """
    root = _project_root_for_scripts_only()
    fused = root / "data" / "enriched_data" / "color_prototypes_fused_lab.csv"
    base = root / "data" / "enriched_data" / "color_prototypes_kmeans.csv"
    return fused if fused.exists() else base


def _default_assignments_for_scripts_only() -> Path:
    """
    Prefer fused assignments if present; otherwise fallback to base assignments.
    """
    root = _project_root_for_scripts_only()
    fused = root / "data" / "enriched_data" / "color_cluster_assignments_fused.csv"
    base = root / "data" / "enriched_data" / "color_cluster_assignments.csv"
    return fused if fused.exists() else base


def _default_outdir_for_scripts_only() -> Path:
    return _project_root_for_scripts_only() / "data" / "scores"


def _default_calibration_for_scripts_only() -> Path:
    return _project_root_for_scripts_only() / "data" / "models" / "color_scoring_calibration.json"


def _default_preference_weights_for_scripts_only() -> Path:
    return _project_root_for_scripts_only() / "data" / "models" / "color_preference_weights.json"


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
    return payload.get("weights")


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

_LEVEL_TO_Q = {
    "low": 0.35,
    "medium": 0.50,
    "high": 0.65,
    "very_high": 0.80,
}


@dataclass(frozen=True)
class Constraint:
    dim: str
    op: str
    level: str
    weight: float = 1.0


@dataclass(frozen=True)
class QuerySpec:
    like_cluster_id: int
    constraints: Tuple[Constraint, ...] = ()


# --- NLP -> scoring constraint adapter ---------------------------------

def _nlp_axis_to_dim(axis: str) -> str | None:
    # Map NLP axes to calibrated scoring dims (must exist in _ALLOWED_DIMS + calibration thresholds)
    if axis == "brightness":
        return "light_hsl"     # calibrated
    if axis == "saturation":
        return "sat_hsl"       # calibrated
    if axis == "vibrancy":
        return "sat_eff"       # calibrated (good proxy for “neon/vibrant”)
    if axis == "depth":
        return "depth"         # calibrated
    if axis == "clarity":
        return "colorfulness"  # proxy (calibrated). refine later if needed
    return None


def _nlp_strength_to_level(direction: str, strength: str) -> str:
    """
    direction: 'raise'|'lower'
    strength: 'weak'|'med'|'strong'
    Output: 'low'|'medium'|'high'|'very_high'
    """
    # Interpretation: for LOWER, strong means "cap earlier" (more restrictive) => medium
    # for RAISE, strong means "require more" => high
    if direction == "lower":
        return {"strong": "medium", "med": "high", "weak": "very_high"}.get(strength, "high")
    return {"strong": "high", "med": "medium", "weak": "low"}.get(strength, "medium")


def _nlp_strength_to_weight(strength: str, *, soft_weight: float) -> float:
    base = {"weak": 0.5, "med": 0.8, "strong": 1.0}.get(strength, 0.8)
    # Treat strong as HARD; downweight non-strong as SOFT
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
        Convert NLP constraints (axis/direction/strength) into scoring constraints (dim/op/level/weight).
        soft_weight downweights non-STRONG constraints globally (SOFT vs HARD).
    """
    out: List[Constraint] = []
    for c in nlp_constraints:
        axis = getattr(getattr(c, "axis", None), "value", None)
        direction = getattr(getattr(c, "direction", None), "value", None)
        strength = getattr(getattr(c, "strength", None), "value", None)
        if not axis or not direction or not strength:
            continue

        dim = _nlp_axis_to_dim(str(axis))
        if dim is None:
            continue

        op = "<=" if str(direction) == "lower" else ">="
        level = _nlp_strength_to_level(str(direction), str(strength))
        weight = _nlp_strength_to_weight(str(strength), soft_weight=float(soft_weight))

        out.append(Constraint(dim=dim, op=op, level=level, weight=weight))
    return tuple(out)


# ---------------------------------------------------------------------
# Public API (stable wrapper)
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

    Example:
      MIX:primary=26;secondary=3;alpha=0.7;C_lab<=high:1.2;sat_eff<=high:0.8

    Returns:
      mix: {"primary": int, "secondary": int, "alpha": float} or None
      constraint_tokens: list of constraint expressions ONLY (no mix params)
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

            # tail can be a param (primary=..) OR already a constraint
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
            # swallow mix params until the first real constraint token appears
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

        # normal constraint outside mix
        out.append(t)

    if mix is None:
        return None, out

    # finalize + validate mix dict
    try:
        primary = int(mix["primary"])
        secondary = int(mix["secondary"])
        alpha = float(mix["alpha"])
    except Exception as e:
        raise ValueError(f"Invalid MIX block in constraints: {constraints!r}") from e

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Invalid MIX alpha={alpha} (expected in [0,1]).")

    return {"primary": primary, "secondary": secondary, "alpha": alpha}, out


def _mix_prototype_rows(
    prototypes: pd.DataFrame,
    *,
    primary: int,
    secondary: int,
    alpha: float,
) -> pd.Series:
    p1 = prototypes.loc[prototypes["cluster_id"] == primary]
    p2 = prototypes.loc[prototypes["cluster_id"] == secondary]
    if p1.empty or p2.empty:
        raise ValueError(f"Unknown MIX clusters: primary={primary}, secondary={secondary}")

    r1 = p1.iloc[0]
    r2 = p2.iloc[0]

    out = r1.copy()
    # Mix in Lab space (required). Also mix C/H if present (for preference term).
    for k in ("L_lab", "a_lab", "b_lab", "C_lab", "H_lab_deg"):
        if k in r1.index and k in r2.index and pd.notna(r1[k]) and pd.notna(r2[k]):
            out[k] = alpha * float(r1[k]) + (1.0 - alpha) * float(r2[k])

    return out


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
    if c.op not in {"<=", ">="}:
        raise ValueError(f"Unknown op '{c.op}'.")
    if c.level not in _LEVEL_TO_Q:
        raise ValueError(f"Unknown level '{c.level}'.")
    if not np.isfinite(c.weight) or c.weight < 0:
        raise ValueError(f"Invalid weight '{c.weight}'.")


# ---------------------------------------------------------------------
# Cluster id injection
# ---------------------------------------------------------------------

def _ensure_cluster_id(
    df: pd.DataFrame,
    prototypes: pd.DataFrame,
    assignments_path: Optional[Path],
) -> pd.DataFrame:
    if "cluster_id" in df.columns:
        return df

    if assignments_path is not None and assignments_path.exists():
        asg = pd.read_csv(assignments_path)
        key_cols = [c for c in ("product_id", "shade_id") if c in df.columns and c in asg.columns]
        if key_cols and "cluster_id" in asg.columns:
            # Ensure merge key dtypes match (pandas merge requires identical dtypes)
            for k in key_cols:
                df[k] = df[k].astype(str)
                asg[k] = asg[k].astype(str)

            merged = df.merge(asg[key_cols + ["cluster_id"]], on=key_cols, how="left")
            if merged["cluster_id"].notna().mean() > 0.95:
                return merged

    _require_cols(df, ("L_lab", "a_lab", "b_lab"), name="enriched df")
    _require_cols(prototypes, ("cluster_id", "L_lab", "a_lab", "b_lab"), name="prototypes")

    X = df[["L_lab", "a_lab", "b_lab"]].to_numpy(float)
    P = prototypes[["L_lab", "a_lab", "b_lab"]].to_numpy(float)
    pid = prototypes["cluster_id"].to_numpy(int)

    nearest = pid[np.argmin(((X[:, None, :] - P[None, :, :]) ** 2).sum(axis=2), axis=1)]
    out = df.copy()
    out["cluster_id"] = nearest
    return out


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
# Constraints (fixed thresholds + normalized penalties)
# ---------------------------------------------------------------------

def _constraint_threshold_fixed(cal: dict, dim: str, level: str) -> float:
    try:
        return float(cal["thresholds"][dim][level])
    except Exception as e:
        raise KeyError(
            f"Missing fixed threshold for dim={dim!r} level={level!r} in calibration."
        ) from e


def _constraint_scale(cal: dict, dim: str) -> float:
    # Prefer IQR; fallback to std; else 1.0 (but this should be rare and visible)
    iqr = float(cal["scale_iqr"].get(dim, float("nan")))
    if np.isfinite(iqr) and iqr > 1e-12:
        return iqr
    std = float(cal["scale_std"].get(dim, float("nan")))
    if np.isfinite(std) and std > 1e-12:
        return std
    return 1.0


def _constraint_penalty(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    if op == "<=":
        return np.maximum(0.0, values - threshold)
    return np.maximum(0.0, threshold - values)


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
    breakdown = {}
    extras = {}

    for i, c in enumerate(constraints, 1):
        _check_constraint(c)
        if c.dim not in work.columns:
            raise KeyError(f"Constraint dim '{c.dim}' missing from work df columns.")
        thr = _constraint_threshold_fixed(calibration, c.dim, c.level)
        scale = _constraint_scale(calibration, c.dim)

        vals = work[c.dim].to_numpy(float)
        gap = _constraint_penalty(vals, c.op, thr)
        gap_norm = gap / scale

        pen = c.weight * gap_norm
        total += pen
        breakdown[f"penalty_{i}__{c.dim}{c.op}{c.level}"] = pen
        extras[f"c{i}__{c.dim}"] = vals

    return total, pd.DataFrame(breakdown, index=work.index), pd.DataFrame(extras, index=work.index)


# ---------------------------------------------------------------------
# Scoring (CORE) - no implicit disk access
# ---------------------------------------------------------------------

def score_shades(
    df: pd.DataFrame,
    prototypes: pd.DataFrame,
    query: QuerySpec,
    *,
    lambda_constraints: float = 1.0,
    lambda_preference: float = 1.0,
    proto_row_override: Optional[pd.Series] = None,
    calibration: dict | None = None,
    preference_weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    CORE scoring function.

    Contract:
      - No implicit file IO.
      - calibration must be provided.
      - if lambda_preference != 0, preference_weights must be provided.
    """
    if calibration is None:
        raise ValueError(
            "Missing 'calibration' dict. Pass it explicitly (loaded via io/assets or load_scoring_calibration(path))."
        )

    _require_cols(
        df,
        ("product_id", "shade_id", "brand_name", "product_name", "shade_name", "url", "L_lab", "a_lab", "b_lab"),
        name="df",
    )

    work = df.dropna(subset=["L_lab", "a_lab", "b_lab"]).copy()
    if work.empty:
        raise ValueError("No valid Lab rows to score.")

    if proto_row_override is not None:
        proto_row = proto_row_override
    else:
        proto = prototypes.loc[prototypes["cluster_id"] == query.like_cluster_id]
        if proto.empty:
            raise ValueError(f"Unknown cluster_id={query.like_cluster_id}")
        proto_row = proto.iloc[0]

    deltaE = _delta_e_to_proto(work, proto_row)
    deltaE_n = _delta_e_norm(deltaE, calibration)

    total_penalty_n, penalty_df, extras_df = _apply_constraints(
        work, query.constraints, calibration=calibration
    )

    if float(lambda_preference) != 0.0:
        if preference_weights is None:
            raise ValueError(
                "lambda_preference != 0 but 'preference_weights' is None. "
                "Pass weights explicitly (load via io/assets or load_preference_weights(path))."
            )

        _require_cols(work, ("C_lab", "H_lab_deg"), name="preference features")
        if "C_lab" not in proto_row.index or "H_lab_deg" not in proto_row.index:
            raise KeyError("proto_row missing required preference columns: ['C_lab', 'H_lab_deg']")

        # normalized deltas
        std_L = _constraint_scale(calibration, "L_lab")
        std_C = _constraint_scale(calibration, "C_lab")

        dL = (work["L_lab"].to_numpy(float) - float(proto_row["L_lab"])) / std_L
        dC = (work["C_lab"].to_numpy(float) - float(proto_row["C_lab"])) / std_C
        dH = (work["H_lab_deg"].to_numpy(float) - float(proto_row["H_lab_deg"]) + 180.0) % 360.0 - 180.0
        dH = dH / 180.0

        preference_score = (
            float(preference_weights["w_L"]) * dL
            + float(preference_weights["w_C"]) * dC
            + float(preference_weights["w_H"]) * dH
        )
    else:
        preference_score = 0.0

    score = (
        -deltaE_n
        + float(lambda_preference) * preference_score
        - float(lambda_constraints) * total_penalty_n
    )

    out = pd.DataFrame(
        {
            "product_id": work["product_id"].astype(str),
            "shade_id": work["shade_id"].astype(str),
            "brand_name": work["brand_name"].astype(str),
            "product_name": work["product_name"].astype(str),
            "shade_name": work["shade_name"].astype(str),
            "url": work["url"].astype(str),
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

    # deterministic due to stable sort + tie-break
    out = out.drop_duplicates(subset=["url", "shade_name"], keep="first")

    out = out.reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


# ---------------------------------------------------------------------
# Script-friendly wrapper (explicit paths; defaults only for scripts)
# ---------------------------------------------------------------------

def score_inventory(
    *,
    inventory: pd.DataFrame,
    prototypes: Optional[pd.DataFrame] = None,
    assignments_path: Optional[str | Path] = None,
    cluster_id: int,
    constraints: str = "",
    lambda_constraints: float = 1.0,
    # IMPORTANT: keep default 0.0 for deterministic goldens unless explicitly enabled
    lambda_preference: float = 0.0,
    calibration_path: Optional[str | Path] = None,
    preference_weights_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Script/test-friendly entrypoint.

    Notes:
      - Still allows script defaults, but they are confined here (NOT in score_shades).
      - For runtime use, prefer calling score_shades with injected assets.
    """
    # Calibration is mandatory; allow script default here only.
    cal_path = Path(calibration_path) if calibration_path is not None else _default_calibration_for_scripts_only()
    cal = load_scoring_calibration(cal_path)

    # auto-pick defaults (prefer fused) - script only
    asg_path = Path(assignments_path) if assignments_path is not None else _default_assignments_for_scripts_only()

    if prototypes is None:
        proto_path = _default_prototypes_for_scripts_only()
        prototypes = pd.read_csv(proto_path)

    df = _ensure_cluster_id(inventory, prototypes, asg_path)

    mix, constraint_tokens = _parse_mix_and_constraints(constraints)

    parsed: List[Constraint] = []
    for s in constraint_tokens:
        parsed.append(_parse_constraint(s))

    query = QuerySpec(like_cluster_id=int(cluster_id), constraints=tuple(parsed))

    proto_override = None
    if mix is not None:
        proto_override = _mix_prototype_rows(
            prototypes,
            primary=int(mix["primary"]),
            secondary=int(mix["secondary"]),
            alpha=float(mix["alpha"]),
        )

    pref_weights = None
    if float(lambda_preference) != 0.0:
        pw_path = (
            Path(preference_weights_path)
            if preference_weights_path is not None
            else _default_preference_weights_for_scripts_only()
        )
        pref_weights = load_preference_weights(pw_path)

    return score_shades(
        df,
        prototypes,
        query,
        lambda_constraints=float(lambda_constraints),
        lambda_preference=float(lambda_preference),
        proto_row_override=proto_override,
        calibration=cal,
        preference_weights=pref_weights,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

_CONSTRAINT_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*(<=|>=)\s*(low|medium|high|very_high)\s*:\s*([0-9]*\.?[0-9]+)\s*$"
)


def _parse_constraint(s: str) -> Constraint:
    raw = s.strip()
    m = _CONSTRAINT_RE.match(raw)
    if not m:
        raise ValueError(f"Invalid constraint '{raw}'.")
    dim, op, level, w_str = m.groups()
    c = Constraint(dim=dim, op=op, level=level, weight=float(w_str))
    _check_constraint(c)
    return c


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--cluster-id", type=int, required=True)
    p.add_argument("--constraint", action="append", default=[])
    p.add_argument("--lambda-constraints", type=float, default=1.0)
    p.add_argument("--lambda-preference", type=float, default=1.0)
    p.add_argument("--infile", type=str, default=str(_default_enriched_for_scripts_only()))
    p.add_argument("--prototypes", type=str, default=str(_default_prototypes_for_scripts_only()))
    p.add_argument("--assignments", type=str, default=str(_default_assignments_for_scripts_only()))
    p.add_argument("--calibration", type=str, default=str(_default_calibration_for_scripts_only()))
    p.add_argument("--preference-weights", type=str, default=str(_default_preference_weights_for_scripts_only()))
    p.add_argument("--outdir", type=str, default=str(_default_outdir_for_scripts_only()))
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    df = pd.read_csv(Path(args.infile))
    prototypes = pd.read_csv(Path(args.prototypes))
    assignments = Path(args.assignments) if args.assignments else None

    df = _ensure_cluster_id(df, prototypes, assignments)

    constraints = tuple(_parse_constraint(s) for s in (args.constraint or []))
    query = QuerySpec(like_cluster_id=int(args.cluster_id), constraints=constraints)

    cal = load_scoring_calibration(Path(args.calibration))

    pref_weights = None
    if float(args.lambda_preference) != 0.0:
        pref_weights = load_preference_weights(Path(args.preference_weights))

    scored = score_shades(
        df,
        prototypes,
        query,
        lambda_constraints=float(args.lambda_constraints),
        lambda_preference=float(args.lambda_preference),
        calibration=cal,
        preference_weights=pref_weights,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"scores_cluster_{int(args.cluster_id)}.csv"
    scored.to_csv(outpath, index=False)
    print(f"[OK] Wrote {outpath}")


if __name__ == "__main__":
    main()
