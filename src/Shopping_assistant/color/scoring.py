# src/Shopping_assistant/color/scoring.py
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths (project conventions)
# ---------------------------------------------------------------------

def _project_root() -> Path:
    # .../src/Shopping_assistant/color/scoring.py -> parents[3] = project root
    return Path(__file__).resolve().parents[3]


def _default_enriched() -> Path:
    return _project_root() / "data" / "enriched_data" / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv"


def _default_prototypes() -> Path:
    return _project_root() / "data" / "enriched_data" / "color_prototypes_kmeans.csv"


def _default_assignments() -> Path:
    return _project_root() / "data" / "enriched_data" / "color_cluster_assignments.csv"


def _default_outdir() -> Path:
    return _project_root() / "data" / "scores"


# ---------------------------------------------------------------------
# Query / Constraint model (generic)
# ---------------------------------------------------------------------

_ALLOWED_DIMS = {
    # core Lab/LCH
    "L_lab",
    "a_lab",
    "b_lab",
    "C_lab",
    "H_lab_deg",
    # derived
    "depth",
    "warmth",
    "sat_eff",
    # optional (if present)
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
    "very_high": 0.80,   # avoid neon / very vivid
}


@dataclass(frozen=True)
class Constraint:
    dim: str                   # e.g. "C_lab"
    op: str                    # "<=" or ">="
    level: str                 # "low" | "medium" | "high" | "very_high"
    weight: float = 1.0        # penalty multiplier


@dataclass(frozen=True)
class QuerySpec:
    like_cluster_id: int
    constraints: Tuple[Constraint, ...] = ()


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _require_cols(df: pd.DataFrame, cols: Sequence[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def _check_constraint(c: Constraint) -> None:
    if c.dim not in _ALLOWED_DIMS:
        raise ValueError(f"Unknown dim '{c.dim}'. Allowed: {sorted(_ALLOWED_DIMS)}")
    if c.op not in _ALLOWED_OPS:
        raise ValueError(f"Unknown op '{c.op}'. Allowed: {sorted(_ALLOWED_OPS)}")
    if c.level not in _LEVEL_TO_Q:
        raise ValueError(f"Unknown level '{c.level}'. Allowed: {sorted(_LEVEL_TO_Q)}")
    if not np.isfinite(c.weight) or c.weight < 0:
        raise ValueError(f"Invalid weight '{c.weight}'. Must be finite and >= 0.")


# ---------------------------------------------------------------------
# Cluster id injection (merge assignments or fallback)
# ---------------------------------------------------------------------

def _ensure_cluster_id(
    df: pd.DataFrame,
    prototypes: pd.DataFrame,
    assignments_path: Optional[Path],
) -> pd.DataFrame:
    if "cluster_id" in df.columns:
        return df

    # Try merge with assignments
    if assignments_path is not None and assignments_path.exists():
        asg = pd.read_csv(assignments_path)

        key_cols: List[str] = []
        if "product_id" in df.columns and "product_id" in asg.columns:
            key_cols.append("product_id")
        if "shade_id" in df.columns and "shade_id" in asg.columns:
            key_cols.append("shade_id")

        if key_cols and "cluster_id" in asg.columns:
            asg = asg[key_cols + ["cluster_id"]].drop_duplicates()
            merged = df.merge(asg, on=key_cols, how="left")
            # if most rows got a cluster_id, keep it
            if merged["cluster_id"].notna().mean() > 0.95:
                return merged

    # Fallback: nearest prototype in Lab (CIE76)
    _require_cols(df, ("L_lab", "a_lab", "b_lab"), name="enriched df")
    _require_cols(prototypes, ("cluster_id", "L_lab", "a_lab", "b_lab"), name="prototypes")

    X = df[["L_lab", "a_lab", "b_lab"]].to_numpy(dtype=np.float64)
    P = prototypes[["L_lab", "a_lab", "b_lab"]].to_numpy(dtype=np.float64)
    pid = prototypes["cluster_id"].to_numpy(dtype=int)

    d2 = ((X[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
    nearest = pid[np.argmin(d2, axis=1)]

    out = df.copy()
    out["cluster_id"] = nearest
    return out


# ---------------------------------------------------------------------
# Color distance (ΔE CIE76) - vectorized
# ---------------------------------------------------------------------

def _delta_e_to_proto(work: pd.DataFrame, proto_row: pd.Series) -> np.ndarray:
    dL = work["L_lab"].to_numpy(dtype=np.float64) - float(proto_row["L_lab"])
    da = work["a_lab"].to_numpy(dtype=np.float64) - float(proto_row["a_lab"])
    db = work["b_lab"].to_numpy(dtype=np.float64) - float(proto_row["b_lab"])
    return np.sqrt(dL * dL + da * da + db * db)


# ---------------------------------------------------------------------
# Generic constraints: thresholds + penalties
# ---------------------------------------------------------------------

def _constraint_threshold(df: pd.DataFrame, dim: str, op: str, level: str) -> float:
    """
    Threshold boundary for 'level' defined as a quantile of the distribution.
    low/medium/high/very_high correspond to q in _LEVEL_TO_Q.
    The operator only affects which side is penalized, not the threshold location.
    """
    q = _LEVEL_TO_Q[level]
    return float(df[dim].quantile(q))


def _constraint_penalty(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    """
    Soft penalty (ReLU) for constraint violations.
      '<='  penalize values above threshold: relu(x - thr)
      '>='  penalize values below threshold: relu(thr - x)
    """
    if op == "<=":
        return np.maximum(0.0, values - threshold)
    return np.maximum(0.0, threshold - values)


def _apply_constraints(
    work: pd.DataFrame,
    constraints: Tuple[Constraint, ...],
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if not constraints:
        zeros = np.zeros(len(work), dtype=np.float64)
        return zeros, pd.DataFrame(index=work.index), pd.DataFrame(index=work.index)

    for c in constraints:
        _check_constraint(c)
        if c.dim not in work.columns:
            raise KeyError(f"Constraint dim '{c.dim}' not in dataset columns.")

    total = np.zeros(len(work), dtype=np.float64)
    breakdown: dict[str, np.ndarray] = {}
    extras: dict[str, np.ndarray] = {}

    for i, c in enumerate(constraints, start=1):
        thr = _constraint_threshold(work, c.dim, c.op, c.level)
        vals = work[c.dim].to_numpy(dtype=np.float64)

        # violation before weight (gap)
        gap = _constraint_penalty(vals, c.op, thr)
        pen_w = float(c.weight) * gap

        total += pen_w

        # readable breakdown
        col = f"penalty_{i}__{c.dim}{c.op}{c.level}__w{c.weight:g}__thr{thr:.4g}"
        breakdown[col] = pen_w

        # debug columns (no weight)
        extras[f"c{i}__{c.dim}"] = vals
        extras[f"c{i}__thr__{c.dim}{c.op}{c.level}"] = np.full(len(vals), thr, dtype=np.float64)
        extras[f"c{i}__gap__{c.dim}{c.op}{c.level}"] = gap

    return total, pd.DataFrame(breakdown, index=work.index), pd.DataFrame(extras, index=work.index)


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------

def score_shades(
    df: pd.DataFrame,
    prototypes: pd.DataFrame,
    query: QuerySpec,
    *,
    lambda_constraints: float = 1.0,
) -> pd.DataFrame:
    _require_cols(
        df,
        ("product_id", "shade_id", "brand_name", "product_name", "shade_name", "url", "L_lab", "a_lab", "b_lab"),
        name="df",
    )
    _require_cols(prototypes, ("cluster_id", "L_lab", "a_lab", "b_lab"), name="prototypes")

    work = df.dropna(subset=["L_lab", "a_lab", "b_lab"]).copy()
    if work.empty:
        raise ValueError("No rows with valid Lab values to score.")

    proto = prototypes.loc[prototypes["cluster_id"] == query.like_cluster_id]
    if proto.empty:
        raise ValueError(f"Unknown cluster_id={query.like_cluster_id} in prototypes.")
    proto_row = proto.iloc[0]

    deltaE = _delta_e_to_proto(work, proto_row)

    total_penalty, penalty_df, extras_df = _apply_constraints(work, query.constraints)

    score = -deltaE - float(lambda_constraints) * total_penalty

    out = pd.DataFrame(
        {
            "product_id": work["product_id"].astype(str),
            "shade_id": work["shade_id"].astype(str),
            "brand_name": work["brand_name"].astype(str),
            "product_name": work["product_name"].astype(str),
            "shade_name": work["shade_name"].astype(str),
            "url": work["url"].astype(str),
            "cluster_id": work["cluster_id"].astype("Int64") if "cluster_id" in work.columns else pd.Series([pd.NA] * len(work)),
            "deltaE": deltaE,
            "constraint_penalty": total_penalty,
            "score": score,
        },
        index=work.index,
    )

    frames = [out]
    if not extras_df.empty:
        frames.append(extras_df)
    if not penalty_df.empty:
        frames.append(penalty_df)
    out = pd.concat(frames, axis=1)

    # -----------------------------------------------------------------
    # IMPORTANT: deterministic order BEFORE any drop_duplicates
    # so "keep=first" is stable and golden tests don't flap.
    # -----------------------------------------------------------------
    sort_cols = ["score"]
    ascending = [False]
    for c in ("product_id", "shade_id"):
        if c in out.columns:
            sort_cols.append(c)
            ascending.append(True)

    out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort")

    # Deduplicate recommendations (avoid same PDP/shade showing multiple times)
    out = out.drop_duplicates(subset=["url", "shade_name"], keep="first")

    # --- de-dup: same PDP + same shade (case-insensitive) ---
    if "url" in out.columns and "shade_name" in out.columns:
        _shade_norm = out["shade_name"].astype(str).str.casefold().str.strip()
        out = out.assign(_shade_norm=_shade_norm)
        out = out.drop_duplicates(subset=["url", "_shade_norm"], keep="first")
        out = out.drop(columns=["_shade_norm"])

    # rank after final deterministic order + dedup
    out = out.reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def score_inventory(
    *,
    inventory: pd.DataFrame,
    prototypes: pd.DataFrame,
    assignments_path: Optional[Path],
    cluster_id: int,
    constraints: str = "",
    lambda_constraints: float = 1.0,
) -> pd.DataFrame:
    """
    Programmatic API equivalent to the CLI.
    `constraints` may be a ';' separated string of DIM<=level:weight.
    """
    inv = _ensure_cluster_id(inventory, prototypes, assignments_path)

    parts = [p.strip() for p in (constraints or "").split(";") if p.strip()]

    cons: List[Constraint] = []
    i = 0
    while i < len(parts):
        part = parts[i]

        # MIX block: swallow "MIX:primary=.." + subsequent "k=v" tokens
        if part.startswith("MIX:"):
            i += 1
            while i < len(parts) and ("=" in parts[i]) and (("<=" not in parts[i]) and (">=" not in parts[i])):
                i += 1
            continue

        cons.append(_parse_constraint(part))
        i += 1

    query = QuerySpec(like_cluster_id=int(cluster_id), constraints=tuple(cons))
    return score_shades(inv, prototypes, query, lambda_constraints=float(lambda_constraints))


# ---------------------------------------------------------------------
# CLI parsing + filesystem-safe output naming
# ---------------------------------------------------------------------

_CONSTRAINT_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*(<=|>=)\s*(low|medium|high|very_high)\s*:\s*([0-9]*\.?[0-9]+)\s*$"
)


def _parse_constraint(s: str) -> Constraint:
    """
    Format:
      DIM<=level:weight  OR  DIM>=level:weight
    Examples:
      L_lab<=high:0.8
      C_lab<=medium:0.6
      warmth>=medium:0.4
    """
    raw = s.strip()
    m = _CONSTRAINT_RE.match(raw)
    if not m:
        raise ValueError(
            f"Invalid constraint '{raw}'. Expected 'DIM<=level:weight' or 'DIM>=level:weight'."
        )

    dim, op, level, w_str = m.groups()
    c = Constraint(dim=dim, op=op, level=level, weight=float(w_str))
    _check_constraint(c)
    return c


def _safe_filename_token(s: str) -> str:
    """
    Windows-safe slug:
      - remove forbidden chars: < > : " / \ | ? *
      - also simplify <=, >= to le/ge
    """
    s = s.replace("<=", "le").replace(">=", "ge")
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score shades using a target color cluster prototype + generic soft constraints (data-driven thresholds)."
    )
    p.add_argument("--cluster-id", type=int, required=True, help="Target cluster_id (from prototypes).")
    p.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="Add a soft constraint: DIM<=level:weight or DIM>=level:weight. Can be repeated.",
    )
    p.add_argument(
        "--lambda-constraints",
        type=float,
        default=1.0,
        help="Global multiplier applied to the summed constraint penalties.",
    )
    p.add_argument("--infile", type=str, default=str(_default_enriched()), help="Enriched CSV input.")
    p.add_argument("--prototypes", type=str, default=str(_default_prototypes()), help="Prototypes CSV.")
    p.add_argument("--assignments", type=str, default=str(_default_assignments()), help="Assignments CSV.")
    p.add_argument("--outdir", type=str, default=str(_default_outdir()), help="Output directory for scoring CSV.")
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    infile = Path(args.infile)
    proto_path = Path(args.prototypes)
    asg_path = Path(args.assignments) if args.assignments else None
    outdir = Path(args.outdir)

    if not infile.exists():
        raise FileNotFoundError(f"infile not found: {infile}")
    if not proto_path.exists():
        raise FileNotFoundError(f"prototypes not found: {proto_path}")
    if asg_path is not None and not asg_path.exists():
        raise FileNotFoundError(f"assignments not found: {asg_path}")

    df = pd.read_csv(infile)
    prototypes = pd.read_csv(proto_path)

    df = _ensure_cluster_id(df, prototypes, asg_path)

    constraints = tuple(_parse_constraint(s) for s in (args.constraint or []))
    query = QuerySpec(like_cluster_id=int(args.cluster_id), constraints=constraints)

    scored = score_shades(
        df,
        prototypes,
        query,
        lambda_constraints=float(args.lambda_constraints),
    )

    outdir.mkdir(parents=True, exist_ok=True)

    # filesystem-safe output name
    if not constraints:
        ctag = "none"
    else:
        parts = [f"{c.dim}_{c.op}_{c.level}_w{c.weight:g}" for c in constraints]
        ctag = "__".join(_safe_filename_token(p) for p in parts)

    outpath = outdir / f"scored_cluster_{int(args.cluster_id)}__{ctag}.csv"
    scored.to_csv(outpath, index=False)

    print(str(outpath))

    # ------------------------------------------------------------------
    # Terminal preview: top-10 results
    # ------------------------------------------------------------------
    TOP_K = 10
    cols = [
        "rank",
        "score",
        "deltaE",
        "constraint_penalty",
        "brand_name",
        "product_name",
        "shade_name",
        "url",
    ]

    # C_lab debug if present (extras add c1__C_lab when constraint #1 is C_lab)
    if "c1__C_lab" in scored.columns:
        cols.insert(4, "c1__C_lab")

    cols = [c for c in cols if c in scored.columns]
    preview = scored.sort_values("rank", ascending=True).head(TOP_K)[cols]

    print("\n" + "=" * 90)
    print("TOP 10 RECOMMENDED LIPSTICKS")
    print("=" * 90)

    for _, r in preview.iterrows():
        c_lab_txt = f" | C_lab={r['c1__C_lab']:.2f}" if "c1__C_lab" in preview.columns else ""
        print(
            f"[#{int(r['rank']):>2}] score={r['score']:.3f} | ΔE={r['deltaE']:.2f} | pen={r['constraint_penalty']:.2f}{c_lab_txt}\n"
            f"    {r.get('brand_name','')} — {r.get('product_name','')} — {r.get('shade_name','')}\n"
            f"    {r.get('url','')}\n"
        )


if __name__ == "__main__":
    main()
