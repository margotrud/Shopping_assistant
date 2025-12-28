# src/Shopping_assistant/color/cluster_colors.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# sklearn is a standard dependency for this kind of project
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def _project_root() -> Path:
    # .../src/Shopping_assistant/color/cluster_colors.py -> project root = parents[3]
    return Path(__file__).resolve().parents[3]


def _default_infile() -> Path:
    # Your enriched CSV produced by enrich_dataset.py
    # You said: output in data/enriched_data/
    # Example expected name: Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv
    return _project_root() / "data" / "enriched_data" / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv"


def _default_outdir() -> Path:
    return _project_root() / "data" / "enriched_data"


def _plots_dir(outdir: Path) -> Path:
    return outdir / "plots"


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ClusterConfig:
    # Features used for clustering
    features: Tuple[str, ...] = ("L_lab", "C_lab", "H_lab_deg")

    # KMeans parameters
    k: int = 30
    random_state: int = 7
    n_init: int = 20

    # Filter rows without features
    drop_missing: bool = True

    # Output naming
    prototypes_filename: str = "color_prototypes_kmeans.csv"
    assignments_filename: str = "color_cluster_assignments.csv"


# ---------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------

def _circularize_hue_deg(h: np.ndarray) -> np.ndarray:
    """
    Hue is circular. Clustering directly on degrees can create edge artifacts around 0/360.
    Fix: represent hue as sin/cos and cluster in that space.
    """
    rad = np.deg2rad(h % 360.0)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1)


def _build_feature_matrix(df: pd.DataFrame, features: Tuple[str, ...]) -> Tuple[np.ndarray, List[str]]:
    """
    Build a clustering matrix with correct treatment for hue:
      - Keep L, C as is
      - Replace H_lab_deg by (cosH, sinH)
    Returns X and column names used.
    """
    used_cols: List[str] = []
    mats: List[np.ndarray] = []

    for f in features:
        if f == "H_lab_deg":
            H = df["H_lab_deg"].to_numpy(dtype=np.float64)
            Hcs = _circularize_hue_deg(H)
            mats.append(Hcs)
            used_cols.extend(["cos_H", "sin_H"])
        else:
            arr = df[f].to_numpy(dtype=np.float64)[:, None]
            mats.append(arr)
            used_cols.append(f)

    X = np.concatenate(mats, axis=1)
    return X, used_cols


def fit_kmeans_color_clusters(df: pd.DataFrame, *, cfg: ClusterConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      prototypes_df: one row per cluster (cluster_id + centroid in original feature space)
      assignments_df: original rows with cluster_id
    """
    missing = df[list(cfg.features)].isna().any(axis=1)
    work = df.loc[~missing].copy() if cfg.drop_missing else df.copy()

    if work.empty:
        raise ValueError("No rows available for clustering after missing-value filtering.")

    # Build feature matrix with circular hue handling
    X, used_cols = _build_feature_matrix(work, cfg.features)

    # Scale for kmeans stability (especially because L and C are not on same scale as cos/sin)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(
        n_clusters=int(cfg.k),
        random_state=int(cfg.random_state),
        n_init=int(cfg.n_init),
    )
    cluster_id = km.fit_predict(Xs).astype(int)
    work["cluster_id"] = cluster_id

    # Build prototypes: compute centroid in interpretable space (L, a, b; plus L,C,H)
    # We compute robust medians for stability.
    # Include:
    # - L_lab, a_lab, b_lab (directly useful for ΔE scoring later)
    # - LCH: L_lab, C_lab, H_lab_deg (summary)
    agg = work.groupby("cluster_id", as_index=False).agg(
        n=("cluster_id", "size"),
        L_lab=("L_lab", "median"),
        a_lab=("a_lab", "median"),
        b_lab=("b_lab", "median"),
        C_lab=("C_lab", "median"),
        H_lab_deg=("H_lab_deg", "median"),
        hue_hsl_deg=("hue_hsl_deg", "median"),
        sat_hsl=("sat_hsl", "median"),
        light_hsl=("light_hsl", "median"),
        depth=("depth", "median"),
        warmth=("warmth", "median"),
        sat_eff=("sat_eff", "median"),
    )

    # Name clusters deterministically (optional: you can rename later)
    agg["name"] = agg["cluster_id"].map(lambda i: f"cluster_{int(i):02d}")

    # Order prototypes by hue then lightness for readability
    prototypes = agg.sort_values(["H_lab_deg", "L_lab"]).reset_index(drop=True)

    # Assignments: keep keys + cluster_id
    keep_cols = [
        "product_id",
        "shade_id",
        "brand_name",
        "product_name",
        "shade_name",
        "chip_hex",
        "chip_rgb",
        "r",
        "g",
        "b",
        "L_lab",
        "a_lab",
        "b_lab",
        "C_lab",
        "H_lab_deg",
        "cluster_id",
    ]
    keep_cols = [c for c in keep_cols if c in work.columns]
    assignments = work[keep_cols].copy()

    return prototypes, assignments


# ---------------------------------------------------------------------
# Plotting (saved to disk; avoids interactive dependencies)
# ---------------------------------------------------------------------

def _save_plots(df: pd.DataFrame, prototypes: pd.DataFrame, outdir: Path) -> None:
    """
    Saves basic diagnostic plots to outdir/plots.
    """
    import matplotlib.pyplot as plt

    pdir = _plots_dir(outdir)
    pdir.mkdir(parents=True, exist_ok=True)

    # 1) L* vs C*
    plt.figure()
    plt.scatter(df["L_lab"], df["C_lab"], s=8, alpha=0.35)
    plt.scatter(prototypes["L_lab"], prototypes["C_lab"], s=60, alpha=0.9)
    plt.xlabel("L* (Lab)")
    plt.ylabel("Chroma (C*)")
    plt.title("L* vs C* (points + cluster medians)")
    plt.tight_layout()
    plt.savefig(pdir / "l_vs_c.png", dpi=160)
    plt.close()

    # 2) Hue vs C*
    plt.figure()
    plt.scatter(df["H_lab_deg"], df["C_lab"], s=8, alpha=0.35)
    plt.scatter(prototypes["H_lab_deg"], prototypes["C_lab"], s=60, alpha=0.9)
    plt.xlabel("Hue (Lab, deg)")
    plt.ylabel("Chroma (C*)")
    plt.title("Hue vs C* (points + cluster medians)")
    plt.tight_layout()
    plt.savefig(pdir / "hue_vs_c.png", dpi=160)
    plt.close()

    # 3) a* vs b*
    plt.figure()
    plt.scatter(df["a_lab"], df["b_lab"], s=8, alpha=0.35)
    plt.scatter(prototypes["a_lab"], prototypes["b_lab"], s=60, alpha=0.9)
    plt.xlabel("a*")
    plt.ylabel("b*")
    plt.title("Lab a* vs b* (points + cluster medians)")
    plt.tight_layout()
    plt.savefig(pdir / "a_vs_b.png", dpi=160)
    plt.close()


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def run_clustering(infile: Path, outdir: Path, *, cfg: ClusterConfig, save_plots: bool) -> Tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)

    # Hard requirement: must have enriched columns
    required = set(cfg.features) | {"a_lab", "b_lab"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Input CSV missing required columns {missing}. "
            "Run enrich_dataset.py first and point to the enriched CSV."
        )

    prototypes, assignments = fit_kmeans_color_clusters(df, cfg=cfg)

    proto_path = outdir / cfg.prototypes_filename
    asg_path = outdir / cfg.assignments_filename

    prototypes.to_csv(proto_path, index=False)
    assignments.to_csv(asg_path, index=False)

    if save_plots:
        # Use non-missing subset for plots to avoid noise
        clean = df.dropna(subset=["L_lab", "C_lab", "H_lab_deg", "a_lab", "b_lab"]).copy()
        _save_plots(clean, prototypes, outdir)

    return proto_path, asg_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cluster lipstick chip colors (Lab/LCH) and export cluster prototypes for dynamic color matching.",
    )
    p.add_argument(
        "--infile",
        type=str,
        default=str(_default_infile()),
        help="Enriched CSV input (must include Lab/LCH columns).",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(_default_outdir()),
        help="Output directory (prototypes + assignments + optional plots).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=30,
        help="Number of clusters.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable saving diagnostic plots.",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    cfg = ClusterConfig(k=int(args.k))
    proto_path, asg_path = run_clustering(
        infile=Path(args.infile),
        outdir=Path(args.outdir),
        cfg=cfg,
        save_plots=not bool(args.no_plots),
    )

    print(str(proto_path))
    print(str(asg_path))


if __name__ == "__main__":
    main()
