# scripts/build_color_scoring_calibration.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from Shopping_assistant.color.scoring import (
    _ALLOWED_DIMS,
    _default_assignments,
    _default_enriched,
    _default_prototypes,
    _ensure_cluster_id,
)


def _project_root() -> Path:
    # .../scripts/build_color_scoring_calibration.py -> parents[1] = project root
    return Path(__file__).resolve().parents[1]


def _default_outpath() -> Path:
    return _project_root() / "data" / "models" / "color_scoring_calibration.json"


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


def build_calibration(
    *,
    enriched_path: Path,
    prototypes_path: Path,
    assignments_path: Path,
    outpath: Path,
    deltaE_ref_q: float = 0.95,
) -> None:
    df = pd.read_csv(enriched_path)
    prototypes = pd.read_csv(prototypes_path)
    assignments = assignments_path if assignments_path.exists() else None

    df = _ensure_cluster_id(df, prototypes, assignments)

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

    # Thresholds for constraint levels are FIXED here (no df.quantile at scoring time)
    level_to_q: Dict[str, float] = {"low": 0.35, "medium": 0.50, "high": 0.65, "very_high": 0.80}

    thresholds: Dict[str, Dict[str, float]] = {}
    scale_iqr: Dict[str, float] = {}
    scale_std: Dict[str, float] = {}

    for dim in sorted(_ALLOWED_DIMS):
        if dim not in df.columns:
            continue

        x = df[dim].dropna()
        if x.empty:
            continue

        thresholds[dim] = {lvl: float(x.quantile(q)) for (lvl, q) in level_to_q.items()}

        _, _, iqr = _iqr_stats(x)
        std = float(x.std(ddof=0))

        # Robust defaults
        scale_iqr[dim] = float(iqr) if np.isfinite(iqr) and iqr > 1e-12 else float("nan")
        scale_std[dim] = float(std) if np.isfinite(std) and std > 1e-12 else float("nan")

    payload = {
        "version": 1,
        "source": {
            "enriched_path": str(enriched_path),
            "prototypes_path": str(prototypes_path),
            "assignments_path": str(assignments_path),
        },
        "deltaE_ref_q": float(deltaE_ref_q),
        "deltaE_ref": deltaE_ref,
        "thresholds": thresholds,           # dim -> level -> threshold
        "scale_iqr": scale_iqr,             # dim -> iqr
        "scale_std": scale_std,             # dim -> std
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
    p.add_argument("--enriched", type=str, default=str(_default_enriched()))
    p.add_argument("--prototypes", type=str, default=str(_default_prototypes()))
    p.add_argument("--assignments", type=str, default=str(_default_assignments()))
    p.add_argument("--out", type=str, default=str(_default_outpath()))
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
