from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAB_COLS = ("L_lab", "a_lab", "b_lab")


def _require_cols(df: pd.DataFrame, cols: Iterable[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}. Available: {list(df.columns)}")


def _coalesce_lab_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has canonical L_lab/a_lab/b_lab columns.
    Handles L_lab_x/L_lab_y or __inv/__asg suffixes.
    Priority by completeness (least NaN).
    """
    out = df.copy()

    for base in LAB_COLS:
        if base in out.columns:
            continue

        candidates = [f"{base}__inv", f"{base}__asg", f"{base}_y", f"{base}_x", base]
        present = [c for c in candidates if c in out.columns]
        if not present:
            continue

        # pick the most complete column
        best = max(present, key=lambda c: out[c].notna().mean())
        out[base] = pd.to_numeric(out[best], errors="coerce")

    return out


def _default_inventory_path() -> Path:
    return PROJECT_ROOT / "data" / "enriched_data" / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv"


def _default_assignments_path() -> Path:
    fused = PROJECT_ROOT / "data" / "enriched_data" / "color_cluster_assignments_fused.csv"
    base = PROJECT_ROOT / "data" / "enriched_data" / "color_cluster_assignments.csv"
    return fused if fused.exists() else base


def _default_out_path() -> Path:
    return PROJECT_ROOT / "data" / "enriched_data" / "color_prototypes_fused_lab.csv"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute Lab prototypes as cluster-wise means.")
    p.add_argument("--inventory", type=Path, default=None)
    p.add_argument("--assignments", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    inventory_path = args.inventory if args.inventory is not None else _default_inventory_path()
    assignments_path = args.assignments if args.assignments is not None else _default_assignments_path()
    out_path = args.out if args.out is not None else _default_out_path()

    if not inventory_path.is_absolute():
        inventory_path = PROJECT_ROOT / inventory_path
    if not assignments_path.is_absolute():
        assignments_path = PROJECT_ROOT / assignments_path
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory not found: {inventory_path}")
    if not assignments_path.exists():
        raise FileNotFoundError(f"Assignments not found: {assignments_path}")

    print(f"[INFO] inventory={inventory_path}")
    print(f"[INFO] assignments={assignments_path}")
    print(f"[INFO] out={out_path}")

    inv = pd.read_csv(inventory_path)
    asg = pd.read_csv(assignments_path)

    # Normalize keys
    for df in (inv, asg):
        df["product_id"] = df["product_id"].astype(str)
        df["shade_id"] = df["shade_id"].astype(str)

    _require_cols(inv, ("product_id", "shade_id"), label="inventory keys")
    _require_cols(asg, ("product_id", "shade_id", "cluster_id"), label="assignments")

    # Merge to bring Lab into assignments
    # (inv carries Lab; asg may also carry Lab; suffixes ensure no collisions)
    merged = asg.merge(
        inv,
        on=["product_id", "shade_id"],
        how="left",
        suffixes=("__asg", "__inv"),
    )
    merged = _coalesce_lab_columns(merged)

    _require_cols(merged, ("cluster_id",) + LAB_COLS, label="merged canonical")

    # Drop rows without Lab
    work = merged.dropna(subset=list(LAB_COLS)).copy()
    if work.empty:
        raise ValueError("No rows with valid Lab after merge. Inventory enrichment is broken.")

    # Exclude outliers if present
    work["cluster_id"] = work["cluster_id"].astype(int)
    work = work.loc[work["cluster_id"] >= 0].copy()

    prototypes = (
        work.groupby("cluster_id", as_index=False)[list(LAB_COLS)]
        .mean()
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prototypes.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} ({len(prototypes)} clusters)")


if __name__ == "__main__":
    main()
