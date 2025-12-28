from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _require_cols(df: pd.DataFrame, cols: List[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label} missing columns: {missing}\n"
            f"{label} columns are: {list(df.columns)}"
        )


def _deltaE76(L1: np.ndarray, a1: np.ndarray, b1: np.ndarray, L2: float, a2: float, b2: float) -> np.ndarray:
    return np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


def _p90(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, 90))


def _load_paths() -> Tuple[Path, Path, Path]:
    inventory_path = (
        PROJECT_ROOT
        / "data"
        / "enriched_data"
        / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv"
    )
    prototypes_path = (
        PROJECT_ROOT
        / "data"
        / "enriched_data"
        / "color_prototypes_kmeans.csv"
    )
    assignments_path = (
        PROJECT_ROOT
        / "data"
        / "enriched_data"
        / "color_cluster_assignments.csv"
    )
    return inventory_path, prototypes_path, assignments_path


def main() -> None:
    inventory_path, prototypes_path, assignments_path = _load_paths()

    inv = pd.read_csv(inventory_path)
    proto = pd.read_csv(prototypes_path)
    asg = pd.read_csv(assignments_path)

    # keys -> strings for safe merges
    for df in (inv, asg):
        df["product_id"] = df["product_id"].astype(str)
        df["shade_id"] = df["shade_id"].astype(str)

    # required schema
    lab_cols = ["L_lab", "a_lab", "b_lab"]
    _require_cols(inv, ["product_id", "shade_id"] + lab_cols, label="inventory")
    _require_cols(asg, ["product_id", "shade_id", "cluster_id"], label="assignments")
    _require_cols(proto, ["cluster_id"] + lab_cols, label="prototypes")

    # merge items with cluster_id
    df = inv.merge(asg[["product_id", "shade_id", "cluster_id"]], on=["product_id", "shade_id"], how="inner")
    df["cluster_id"] = df["cluster_id"].astype(int)

    # index prototypes
    proto_idx = proto.set_index("cluster_id")[lab_cols]

    rows = []
    for cid, g in df.groupby("cluster_id"):
        n = int(len(g))

        # cluster stats
        L = g["L_lab"].to_numpy(float)
        a = g["a_lab"].to_numpy(float)
        b = g["b_lab"].to_numpy(float)

        mean_L = float(np.nanmean(L))
        mean_a = float(np.nanmean(a))
        mean_b = float(np.nanmean(b))

        std_L = float(np.nanstd(L))
        std_a = float(np.nanstd(a))
        std_b = float(np.nanstd(b))

        if cid in proto_idx.index:
            p = proto_idx.loc[cid]
            pL, pa, pb = float(p["L_lab"]), float(p["a_lab"]), float(p["b_lab"])

            # intra distance = item -> prototype (ΔE76 on Lab)
            de = _deltaE76(L, a, b, pL, pa, pb)
            mean_intra = float(np.nanmean(de))
            p90_intra = _p90(de)

            # prototype vs mean cluster center
            de_proto_vs_mean = float(np.sqrt((pL - mean_L) ** 2 + (pa - mean_a) ** 2 + (pb - mean_b) ** 2))
        else:
            pL = pa = pb = float("nan")
            mean_intra = p90_intra = de_proto_vs_mean = float("nan")

        rows.append({
            "cluster_id": int(cid),
            "n_candidates": n,

            "proto_L": pL,
            "proto_a": pa,
            "proto_b": pb,

            "mean_L": mean_L,
            "mean_a": mean_a,
            "mean_b": mean_b,

            "std_L": std_L,
            "std_a": std_a,
            "std_b": std_b,

            "mean_intra_deltaE76": mean_intra,
            "p90_intra_deltaE76": p90_intra,
            "deltaE76_proto_vs_mean": de_proto_vs_mean,
        })

    out = pd.DataFrame(rows).sort_values(["n_candidates"], ascending=False).reset_index(drop=True)

    outdir = PROJECT_ROOT / "data" / "reports"
    outdir.mkdir(parents=True, exist_ok=True)

    out_path = outdir / "diag_color_clusters.csv"
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
