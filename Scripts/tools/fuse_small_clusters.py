from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# Config (tune here)
# -----------------------------
MIN_CLUSTER_SIZE = 20
OUTLIER_CLUSTER_ID = -1

# Max distance (ΔE76 in Lab) allowed to reassign an item to a healthy cluster prototype.
# If too strict -> too many outliers. If too loose -> you smear cluster semantics.
MAX_ITEM_TO_TARGET_DE = 18.0

# Optional: max distance allowed between SMALL-cluster center and target prototype center.
# Prevents absurd merges. If violated -> all items become outliers.
MAX_CENTER_TO_TARGET_DE = 25.0


# -----------------------------
# Helpers
# -----------------------------
def _require_cols(df: pd.DataFrame, cols: List[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}. Available: {list(df.columns)}")


def _deltaE76_vec(L: np.ndarray, a: np.ndarray, b: np.ndarray, L2: float, a2: float, b2: float) -> np.ndarray:
    return np.sqrt((L - L2) ** 2 + (a - a2) ** 2 + (b - b2) ** 2)


def _deltaE76_point(L1: float, a1: float, b1: float, L2: float, a2: float, b2: float) -> float:
    return float(np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2))


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


@dataclass(frozen=True)
class Proto:
    L: float
    a: float
    b: float


def main() -> None:
    inventory_path, prototypes_path, assignments_path = _load_paths()

    inv = pd.read_csv(inventory_path)
    proto = pd.read_csv(prototypes_path)
    asg = pd.read_csv(assignments_path)

    # keys -> strings for safe merges
    for df in (inv, asg):
        df["product_id"] = df["product_id"].astype(str)
        df["shade_id"] = df["shade_id"].astype(str)

    lab_cols = ["L_lab", "a_lab", "b_lab"]
    _require_cols(inv, ["product_id", "shade_id"] + lab_cols, label="inventory")
    _require_cols(asg, ["product_id", "shade_id", "cluster_id"], label="assignments")
    _require_cols(proto, ["cluster_id"] + lab_cols, label="prototypes")

    # Ensure cluster_id int
    asg["cluster_id"] = asg["cluster_id"].astype(int)
    proto["cluster_id"] = proto["cluster_id"].astype(int)

    # Candidate counts
    counts = (
        asg["cluster_id"]
        .value_counts()
        .rename_axis("cluster_id")
        .reset_index(name="n_candidates")
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )

    healthy = counts.loc[counts["n_candidates"] >= MIN_CLUSTER_SIZE, "cluster_id"].astype(int).tolist()
    small = counts.loc[counts["n_candidates"] < MIN_CLUSTER_SIZE, "cluster_id"].astype(int).tolist()

    if not healthy:
        raise ValueError(f"No healthy clusters found with MIN_CLUSTER_SIZE={MIN_CLUSTER_SIZE}")

    # Prototypes dict for fast lookup
    proto_idx = proto.set_index("cluster_id")[lab_cols]
    protos: Dict[int, Proto] = {}
    for cid in healthy:
        if cid not in proto_idx.index:
            raise KeyError(f"Healthy cluster_id={cid} missing from prototypes file")
        row = proto_idx.loc[cid]
        protos[cid] = Proto(float(row["L_lab"]), float(row["a_lab"]), float(row["b_lab"]))

    # Merge assignments with inventory Lab (to operate item-level)
    df = asg.merge(
        inv[["product_id", "shade_id"] + lab_cols],
        on=["product_id", "shade_id"],
        how="left",
        suffixes=("_asg", "_inv"),
    )

    # Normalize Lab columns after merge (handle _x/_y or custom suffixes)
    # Prefer inventory Lab if present; fallback to assignment-side Lab.
    def _pick_lab(base: str) -> pd.Series:
        # try explicit suffixes first
        inv_col = f"{base}_inv"
        asg_col = f"{base}_asg"
        if inv_col in df.columns and asg_col in df.columns:
            return df[inv_col].where(df[inv_col].notna(), df[asg_col])
        if inv_col in df.columns:
            return df[inv_col]
        if asg_col in df.columns:
            return df[asg_col]

        # fallback to pandas default suffixes if script was edited elsewhere
        y_col = f"{base}_y"
        x_col = f"{base}_x"
        if y_col in df.columns and x_col in df.columns:
            return df[y_col].where(df[y_col].notna(), df[x_col])
        if y_col in df.columns:
            return df[y_col]
        if x_col in df.columns:
            return df[x_col]

        raise KeyError(f"Cannot find Lab column for '{base}'. Available: {list(df.columns)}")

    df["L_lab"] = _pick_lab("L_lab").astype(float)
    df["a_lab"] = _pick_lab("a_lab").astype(float)
    df["b_lab"] = _pick_lab("b_lab").astype(float)

    # Now enforce canonical schema
    _require_cols(df, ["L_lab", "a_lab", "b_lab"], label="asg+inventory merge canonical")

    if df[["L_lab", "a_lab", "b_lab"]].isna().any().any():
        bad = df[df[["L_lab", "a_lab", "b_lab"]].isna().any(axis=1)].head(20)
        raise ValueError(
            "Some assigned items have missing canonical Lab after merge. "
            "Fix inventory enrichment first. Example rows:\n"
            f"{bad[['product_id', 'shade_id', 'cluster_id']].to_string(index=False)}"
        )

    # Precompute centers for each cluster (from items, not from prototype)
    centers = (
        df.groupby("cluster_id", as_index=False)[lab_cols]
        .mean()
        .rename(columns={"L_lab": "center_L", "a_lab": "center_a", "b_lab": "center_b"})
    )

    centers_idx = centers.set_index("cluster_id")

    # Determine target healthy cluster per small cluster using center-to-proto distance
    small_to_target: Dict[int, int] = {}
    small_to_center_de: Dict[int, float] = {}

    for cid in small:
        if cid not in centers_idx.index:
            continue
        cL = float(centers_idx.loc[cid, "center_L"])
        ca = float(centers_idx.loc[cid, "center_a"])
        cb = float(centers_idx.loc[cid, "center_b"])

        best_target = None
        best_de = float("inf")
        for hid in healthy:
            p = protos[hid]
            de = _deltaE76_point(cL, ca, cb, p.L, p.a, p.b)
            if de < best_de:
                best_de = de
                best_target = hid

        assert best_target is not None
        small_to_target[cid] = int(best_target)
        small_to_center_de[cid] = float(best_de)

    # Reassign items
    new_cluster = df["cluster_id"].copy()
    report_rows: List[Dict[str, object]] = []

    for cid in small:
        if cid not in small_to_target:
            continue

        target = small_to_target[cid]
        center_de = small_to_center_de[cid]
        n = int((df["cluster_id"] == cid).sum())

        p = protos[target]

        # If the whole cluster center is absurdly far from any healthy cluster -> mark all as outliers
        if center_de > MAX_CENTER_TO_TARGET_DE:
            new_cluster.loc[df["cluster_id"] == cid] = OUTLIER_CLUSTER_ID
            report_rows.append({
                "small_cluster_id": cid,
                "n_items": n,
                "target_cluster_id": target,
                "center_to_target_de": center_de,
                "status": "CENTER_TOO_FAR_ALL_OUTLIERS",
                "n_reassigned": 0,
                "n_outliers": n,
                "mean_item_to_target_de": float("nan"),
                "p90_item_to_target_de": float("nan"),
            })
            continue

        mask = df["cluster_id"] == cid
        L = df.loc[mask, "L_lab"].to_numpy(float)
        a = df.loc[mask, "a_lab"].to_numpy(float)
        b = df.loc[mask, "b_lab"].to_numpy(float)

        de_items = _deltaE76_vec(L, a, b, p.L, p.a, p.b)

        ok = de_items <= MAX_ITEM_TO_TARGET_DE
        n_ok = int(ok.sum())
        n_bad = int((~ok).sum())

        # Apply per-item reassignment/outlier
        idx = df.index[mask]
        new_cluster.loc[idx[ok]] = target
        new_cluster.loc[idx[~ok]] = OUTLIER_CLUSTER_ID

        report_rows.append({
            "small_cluster_id": cid,
            "n_items": n,
            "target_cluster_id": target,
            "center_to_target_de": center_de,
            "status": "MERGED_WITH_ITEM_GATE",
            "n_reassigned": n_ok,
            "n_outliers": n_bad,
            "mean_item_to_target_de": float(np.mean(de_items)),
            "p90_item_to_target_de": float(np.percentile(de_items, 90)),
        })

    # Write fused assignments
    out_asg = asg.copy()
    out_asg["cluster_id"] = new_cluster.astype(int)

    outdir_reports = PROJECT_ROOT / "data" / "reports"
    outdir_reports.mkdir(parents=True, exist_ok=True)

    outdir_enriched = PROJECT_ROOT / "data" / "enriched_data"
    outdir_enriched.mkdir(parents=True, exist_ok=True)

    fused_path = outdir_enriched / "color_cluster_assignments_fused.csv"
    out_asg.to_csv(fused_path, index=False)

    # Report
    rep = pd.DataFrame(report_rows).sort_values(["n_items"], ascending=False).reset_index(drop=True)
    rep_path = outdir_reports / "fuse_small_clusters_report.csv"
    rep.to_csv(rep_path, index=False)

    # Summary counts after fusion
    counts_after = (
        out_asg["cluster_id"]
        .value_counts()
        .rename_axis("cluster_id")
        .reset_index(name="n_candidates_after")
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )
    counts_after_path = outdir_reports / "cluster_counts_after_fusion.csv"
    counts_after.to_csv(counts_after_path, index=False)

    print(f"[OK] wrote {fused_path}")
    print(f"[OK] wrote {rep_path}")
    print(f"[OK] wrote {counts_after_path}")

    print(
        f"[INFO] healthy_clusters={len(healthy)} small_clusters={len(small)} "
        f"MIN_CLUSTER_SIZE={MIN_CLUSTER_SIZE} MAX_ITEM_TO_TARGET_DE={MAX_ITEM_TO_TARGET_DE} "
        f"MAX_CENTER_TO_TARGET_DE={MAX_CENTER_TO_TARGET_DE} OUTLIER_CLUSTER_ID={OUTLIER_CLUSTER_ID}"
    )


if __name__ == "__main__":
    main()
