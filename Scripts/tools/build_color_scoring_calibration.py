# scripts/tools/build_color_scoring_calibration.py
"""
OFFLINE SCRIPT — NOT RUN IN THIS REPOSITORY

This script rebuilds `data/models/color_scoring_calibration.json`
from private enriched datasets that are NOT versioned.

It is kept for reproducibility and documentation purposes only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def _project_root() -> Path:
    # .../scripts/tools/build_color_scoring_calibration.py -> parents[2] = project root
    return Path(__file__).resolve().parents[2]


def _default_outpath() -> Path:
    return _project_root() / "data" / "models" / "color_scoring_calibration.json"


def _default_enriched() -> Path:
    return _project_root() / "data" / "enriched_data" / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv"


def _default_prototypes() -> Path:
    return _project_root() / "data" / "enriched_data" / "color_prototypes_kmeans.csv"


def _default_assignments() -> Path:
    return _project_root() / "data" / "enriched_data" / "color_cluster_assignments.csv"


def _require_cols(df: pd.DataFrame, cols: Iterable[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def _iqr_stats(x: pd.Series) -> Tuple[float, float, float]:
    x = x.dropna()
    if x.empty:
        return float("nan"), float("nan"), float("nan")
    q25 = float(x.quantile(0.25))
    q75 = float(x.quantile(0.75))
    iqr = float(q75 - q25)
    return q25, q75, iqr


def _relpath_under_root(p: Path, *, root: Path) -> str:
    """
    Repo-portable metadata string:
    - if p under root: POSIX relative path
    - else: filename only (avoid leaking local absolute paths)
    """
    try:
        return str(p.resolve().relative_to(root).as_posix())
    except Exception:
        return p.name


def _allowed_dims_from_df(df: pd.DataFrame) -> list[str]:
    """
    Avoid importing internal constants from scoring.py.
    Infer dims from columns present in df using a conservative allowlist.
    """
    candidate_dims = [
        "brightness",
        "depth",
        "saturation",
        "vibrancy",
        "chroma",
        "clarity",
        "warmth",
    ]
    return [d for d in candidate_dims if d in df.columns]


def _ensure_cluster_id_local(
    df: pd.DataFrame,
    prototypes: pd.DataFrame,
    assignments: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Ensure df has 'cluster_id'.
    Strategy:
      1) if df already has cluster_id -> ok
      2) else if assignments provided -> merge on best available key
      3) else: raise (cannot infer)
    """
    if "cluster_id" in df.columns and df["cluster_id"].notna().any():
        return df

    if assignments is None:
        raise ValueError(
            "No 'cluster_id' in enriched df and no assignments provided. "
            "Provide --assignments with columns including cluster_id + a join key (e.g. shade_id)."
        )

    if "cluster_id" not in assignments.columns:
        raise KeyError("assignments missing required column: cluster_id")

    # choose join key among common identifiers
    join_candidates = ["shade_id", "product_id", "sku", "item_id"]
    join_key = None
    for k in join_candidates:
        if k in df.columns and k in assignments.columns:
            join_key = k
            break

    if join_key is None:
        raise KeyError(
            f"Cannot merge assignments: no common join key found. "
            f"Tried {join_candidates}. df_cols={sorted(df.columns)[:20]}..."
        )

    merged = df.merge(assignments[[join_key, "cluster_id"]], on=join_key, how="left")
    if "cluster_id" not in merged.columns or merged["cluster_id"].isna().all():
        raise ValueError(f"Assignments merge failed: cluster_id still missing after join on '{join_key}'")

    # basic sanity: cluster ids should exist in prototypes
    if "cluster_id" in prototypes.columns:
        # no hard fail: merge-to-prototypes later will naturally drop bad rows
        pass

    return merged


def build_calibration(
    *,
    enriched_path: Path,
    prototypes_path: Path,
    assignments_path: Path,
    outpath: Path,
    deltaE_ref_q: float = 0.95,
) -> None:
    # fail fast with an explicit message (repo is lightweight; these files may be absent)
    if not enriched_path.exists():
        raise FileNotFoundError(
            f"Enriched CSV not found: {enriched_path}\n"
            f"Provide --enriched or set SA_ENRICHED_CSV_PATH."
        )
    if not prototypes_path.exists():
        raise FileNotFoundError(
            f"Prototypes CSV not found: {prototypes_path}\n"
            f"Provide --prototypes or set SA_PROTOTYPES_CSV_PATH."
        )

    df = pd.read_csv(enriched_path)
    prototypes = pd.read_csv(prototypes_path)

    assignments_df: Optional[pd.DataFrame] = None
    if assignments_path and Path(assignments_path).exists():
        assignments_df = pd.read_csv(assignments_path)

    df = _ensure_cluster_id_local(df, prototypes, assignments_df)

    # Needed for deltaE-to-own-prototype distribution
    _require_cols(df, ("cluster_id", "L_lab", "a_lab", "b_lab"), name="enriched df")
    _require_cols(prototypes, ("cluster_id", "L_lab", "a_lab", "b_lab"), name="prototypes")

    # Merge each item with its cluster prototype
    proto_cols = ["cluster_id", "L_lab", "a_lab", "b_lab", "C_lab", "H_lab_deg"]
    proto_cols = [c for c in proto_cols if c in prototypes.columns]
    P = prototypes[proto_cols].copy()

    merged = df.merge(P, on="cluster_id", how="left", suffixes=("", "__proto"))
    merged = merged.dropna(subset=["L_lab", "a_lab", "b_lab", "L_lab__proto", "a_lab__proto", "b_lab__proto"])

    dL = merged["L_lab"].to_numpy(float) - merged["L_lab__proto"].to_numpy(float)
    da = merged["a_lab"].to_numpy(float) - merged["a_lab__proto"].to_numpy(float)
    db = merged["b_lab"].to_numpy(float) - merged["b_lab__proto"].to_numpy(float)
    deltaE = np.sqrt(dL * dL + da * da + db * db)

    if len(deltaE) == 0:
        raise ValueError("Cannot build calibration: no valid deltaE rows (check enriched/prototypes).")

    deltaE_ref = float(np.quantile(deltaE, float(deltaE_ref_q)))
    if not np.isfinite(deltaE_ref) or deltaE_ref <= 0:
        raise ValueError(f"Invalid deltaE_ref={deltaE_ref}")

    level_to_q: Dict[str, float] = {"low": 0.35, "medium": 0.50, "high": 0.65, "very_high": 0.80}

    thresholds: Dict[str, Dict[str, float]] = {}
    scale_iqr: Dict[str, float] = {}
    scale_std: Dict[str, float] = {}

    for dim in _allowed_dims_from_df(df):
        x = df[dim].dropna()
        if x.empty:
            continue

        thresholds[dim] = {lvl: float(x.quantile(q)) for (lvl, q) in level_to_q.items()}

        _, _, iqr = _iqr_stats(x)
        std = float(x.std(ddof=0))

        scale_iqr[dim] = float(iqr) if np.isfinite(iqr) and iqr > 1e-12 else float("nan")
        scale_std[dim] = float(std) if np.isfinite(std) and std > 1e-12 else float("nan")

    root = _project_root().resolve()
    payload = {
        "version": 1,
        "source": {
            "enriched_path": _relpath_under_root(enriched_path, root=root),
            "prototypes_path": _relpath_under_root(prototypes_path, root=root),
            "assignments_path": _relpath_under_root(assignments_path, root=root),
        },
        "deltaE_ref_q": float(deltaE_ref_q),
        "deltaE_ref": deltaE_ref,
        "thresholds": thresholds,
        "scale_iqr": scale_iqr,
        "scale_std": scale_std,
        "notes": {
            "constraint_thresholds": "Fixed quantiles computed on reference dataset; DO NOT recompute at scoring time.",
            "penalty_normalization": "gap / scale_iqr[dim] (fallback scale_std).",
            "deltaE_normalization": "deltaE / deltaE_ref",
        },
    }

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote {outpath}")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--enriched", type=str, default=os.environ.get("SA_ENRICHED_CSV_PATH", str(_default_enriched())))
    p.add_argument("--prototypes", type=str, default=os.environ.get("SA_PROTOTYPES_CSV_PATH", str(_default_prototypes())))
    p.add_argument("--assignments", type=str, default=os.environ.get("SA_ASSIGNMENTS_CSV_PATH", str(_default_assignments())))
    p.add_argument("--out", type=str, default=os.environ.get("SA_CALIBRATION_JSON_PATH", str(_default_outpath())))
    p.add_argument("--deltaE-ref-q", type=float, default=0.95)
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    build_calibration(
        enriched_path=Path(args.enriched),
        prototypes_path=Path(args.prototypes),
        assignments_path=Path(args.assignments),
        outpath=Path(args.out),
        deltaE_ref_q=float(args.deltaE_ref_q),
    )


if __name__ == "__main__":
    main()
