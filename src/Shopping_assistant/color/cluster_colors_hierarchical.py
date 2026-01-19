# src/Shopping_assistant/color/cluster_colors_hierarchical.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def _project_root() -> Path:
    # .../src/Shopping_assistant/color/cluster_colors_hierarchical.py -> project root = parents[3]
    return Path(__file__).resolve().parents[3]


def _default_infile() -> Path:
    # Output of enrich_dataset.py
    return _project_root() / "data" / "enriched_data" / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv"


def _default_outdir() -> Path:
    return _project_root() / "data" / "enriched_data"


def _plots_dir(outdir: Path) -> Path:
    return outdir / "plots"


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class HierClusterConfig:
    # Level-1 (family) features: circular hue + (C,L) for separation
    family_features: Tuple[str, ...] = ("cos_H", "sin_H", "C_lab", "L_lab")
    # Applied AFTER standardization (keeps scales comparable, avoids duplicated columns)
    family_feature_weights: Tuple[float, ...] = (2.0, 2.0, 1.0, 1.0)
    family_k_min: int = 4
    family_k_max: int = 12

    # Post-fit family cleanup: avoid tiny GMM families by merging to nearest family prototype
    min_family_samples_after_fit: int = 80
    max_family_merge_passes: int = 5

    # Family refinement (split overly broad families by tone)
    # NOTE: The "safe split" threshold must be DECOUPLED from min_family_samples_after_fit.
    family_de_p90_max: float = 30.0
    max_family_refine_passes: int = 8
    family_refine_features: Tuple[str, ...] = ("L_lab", "C_lab")  # tone split; deterministic

    # refinement safety
    # - split is allowed only if each child has >= min_family_child_samples_for_refine
    # - refinement will SKIP families that cannot be safely split, instead of stopping.
    min_family_child_samples_for_refine: int = 40
    refine_try_topk_worst_families: int = 8  # per pass: attempt among top-K worst before giving up

    # Level-2 (subcluster) features within family (GMM on standardized Lab+C)
    sub_features: Tuple[str, ...] = ("L_lab", "a_lab", "b_lab", "C_lab")
    sub_k_max: int = 6

    # Robustness / stability
    random_state: int = 7
    max_iter: int = 500
    n_init: int = 3

    # Minimum samples to attempt a split; else force 1 subcluster
    min_samples_per_family_for_split: int = 60
    min_samples_per_subcluster: int = 40  # CONTRACT: no final subcluster below this

    # Post-fit subcluster cleanup: merge tiny subclusters to nearest big subcluster prototype (WITHIN family)
    max_subcluster_merge_passes: int = 5

    # -----------------------------------------------------------------
    # Post-pass cluster repair (fix overwide clusters after subclustering)
    # -----------------------------------------------------------------
    enable_cluster_repair: bool = True
    cluster_de_p90_max: float = 30.0
    cluster_hue_span_deg_for_hue_split: float = 60.0

    # IMPORTANT: default must respect the global contract (min_samples_per_subcluster).
    # Repair is not allowed to create children smaller than min_samples_per_subcluster.
    min_cluster_child_samples_for_repair: int = 40

    max_cluster_repair_splits: int = 8

    # IMPORTANT FIX: try top-K worst clusters per pass (avoid looping on same unsplittable worst cluster)
    repair_try_topk_worst_clusters: int = 8

    # Outputs
    prototypes_filename: str = "color_prototypes_hier.csv"
    assignments_filename: str = "color_cluster_assignments_hier.csv"
    report_filename: str = "cluster_report_hier.json"

    # Optional naming via quantiles (deterministic, explainable)
    enable_auto_labels: bool = True


# ---------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------

def _circularize_hue_deg(h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rad = np.deg2rad(h % 360.0)
    return np.cos(rad), np.sin(rad)


def _ensure_family_basis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures df has cos_H / sin_H derived from H_lab_deg.
    """
    if "cos_H" in df.columns and "sin_H" in df.columns:
        return df
    if "H_lab_deg" not in df.columns:
        raise KeyError("Missing H_lab_deg (required for circular hue features).")
    cos_h, sin_h = _circularize_hue_deg(df["H_lab_deg"].to_numpy(dtype=np.float64))
    out = df.copy()
    out["cos_H"] = cos_h
    out["sin_H"] = sin_h
    return out


def _matrix(df: pd.DataFrame, cols: Tuple[str, ...]) -> np.ndarray:
    return df[list(cols)].to_numpy(dtype=np.float64)


def _apply_feature_weights(Xs: np.ndarray, weights: Tuple[float, ...]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if Xs.ndim != 2:
        raise ValueError(f"Xs must be 2D, got shape={Xs.shape}")
    if Xs.shape[1] != w.shape[0]:
        raise ValueError(f"weights len={w.shape[0]} must match n_features={Xs.shape[1]}")
    return Xs * w


def _de76(L1: np.ndarray, a1: np.ndarray, b1: np.ndarray, L2: float, a2: float, b2: float) -> np.ndarray:
    return np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


# ---------------------------------------------------------------------
# Model selection (BIC)
# ---------------------------------------------------------------------

def _fit_gmm_bic(
    Xs: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    random_state: int,
    max_iter: int,
    n_init: int,
) -> Tuple[GaussianMixture, List[Dict[str, float]]]:
    """
    Select number of components via BIC over [k_min..k_max], return best model + report rows.
    Uses full covariance for flexibility in color space.
    """
    rows: List[Dict[str, float]] = []
    best: Optional[GaussianMixture] = None
    best_bic: float = float("inf")

    for k in range(int(k_min), int(k_max) + 1):
        gmm = GaussianMixture(
            n_components=int(k),
            covariance_type="full",
            random_state=int(random_state),
            max_iter=int(max_iter),
            n_init=int(n_init),
            reg_covar=1e-6,
        )
        gmm.fit(Xs)
        bic = float(gmm.bic(Xs))
        aic = float(gmm.aic(Xs))
        rows.append({"k": float(k), "bic": bic, "aic": aic})
        if bic < best_bic:
            best_bic = bic
            best = gmm

    assert best is not None
    return best, rows


# ---------------------------------------------------------------------
# Post-fit family cleanup (merge tiny families)
# ---------------------------------------------------------------------

def _merge_small_families_once(
    work: pd.DataFrame,
    *,
    family_features: Tuple[str, ...],
    family_feature_weights: Tuple[float, ...],
    min_family_n: int,
) -> Tuple[pd.DataFrame, dict]:
    """
    Merge any family with n < min_family_n to the nearest BIG family prototype
    (big = n >= min_family_n) in standardized+weighted family feature space.
    One pass only.
    """
    counts = work["family_id"].value_counts()
    small = counts[counts < int(min_family_n)].index.tolist()
    big = counts[counts >= int(min_family_n)].index.tolist()

    info = {
        "min_family_n": int(min_family_n),
        "n_families_before": int(counts.size),
        "small_families": [int(x) for x in small],
        "big_families": [int(x) for x in big],
        "small_total_n": int(counts.loc[small].sum()) if small else 0,
    }

    if (not small) or (not big):
        return work, {**info, "changed": False, "mode": "no_merge"}

    prot = (
        work[work["family_id"].isin(big)]
        .groupby("family_id", as_index=False)[list(family_features)]
        .median()
        .sort_values("family_id")
        .reset_index(drop=True)
    )

    sc = StandardScaler()
    Xs = sc.fit_transform(work[list(family_features)].to_numpy(float))
    Ps = sc.transform(prot[list(family_features)].to_numpy(float))

    Xs = _apply_feature_weights(Xs, family_feature_weights)
    Ps = _apply_feature_weights(Ps, family_feature_weights)

    fam_ids = prot["family_id"].to_numpy(int)
    d2 = ((Xs[:, None, :] - Ps[None, :, :]) ** 2).sum(axis=2)
    nearest_big = fam_ids[np.argmin(d2, axis=1)]

    out = work.copy()
    mask_small = out["family_id"].isin(small).to_numpy()
    out.loc[mask_small, "family_id"] = nearest_big[mask_small]

    return out, {**info, "changed": True, "mode": "merged_to_big_targets"}


def _compact_ids(s: pd.Series) -> pd.Series:
    vals = s.astype(int).to_numpy()
    uniq = sorted(set(vals.tolist()))
    remap = {v: i for i, v in enumerate(uniq)}
    return pd.Series([remap[v] for v in vals], index=s.index, dtype="int64")


def _compact_subcluster_ids_within_family(work: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure subcluster_id is compact (0..K-1) within each family.
    This is important after repair splits (which may create gaps).
    """
    out = work.copy()
    for fam in sorted(out["family_id"].unique().tolist()):
        mask = out["family_id"] == fam
        out.loc[mask, "subcluster_id"] = _compact_ids(out.loc[mask, "subcluster_id"])
    return out


# ---------------------------------------------------------------------
# Family refinement (split overly broad families by tone)
# ---------------------------------------------------------------------

def _family_de_p90_to_proto(work: pd.DataFrame, fam_id: int) -> float:
    fam = work[work["family_id"] == fam_id]
    if fam.empty:
        return float("nan")
    L0 = float(fam["L_lab"].median())
    a0 = float(fam["a_lab"].median())
    b0 = float(fam["b_lab"].median())
    d = _de76(
        fam["L_lab"].to_numpy(float),
        fam["a_lab"].to_numpy(float),
        fam["b_lab"].to_numpy(float),
        L0, a0, b0,
    )
    return float(np.quantile(d, 0.9))


def _split_family_once_by_tone(
    work: pd.DataFrame,
    *,
    fam_id: int,
    cfg: HierClusterConfig,
) -> Tuple[pd.DataFrame, dict]:
    """
    Attempt a single split of one family into 2 parts using GMM(k=2) over cfg.family_refine_features.

    Safety constraints (REFINE ONLY):
      - n must be >= 2 * cfg.min_family_child_samples_for_refine
      - both children must have >= cfg.min_family_child_samples_for_refine
    """
    fam = work[work["family_id"] == fam_id].copy()
    n = int(len(fam))

    info = {
        "family_id": int(fam_id),
        "n": n,
        "mode": "no_split",
        "changed": False,
    }

    min_child = int(cfg.min_family_child_samples_for_refine)
    if n < 2 * min_child:
        return work, {**info, "reason": "not_enough_samples_for_safe_split", "min_child": min_child}

    feats = cfg.family_refine_features
    missing = [c for c in feats if c not in fam.columns]
    if missing:
        return work, {**info, "reason": "missing_refine_features", "missing": missing}

    X = fam[list(feats)].to_numpy(float)
    Xs = StandardScaler().fit_transform(X)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=int(cfg.random_state),
        max_iter=int(cfg.max_iter),
        n_init=int(cfg.n_init),
        reg_covar=1e-6,
    )
    gmm.fit(Xs)
    y = gmm.predict(Xs).astype(int)

    c0 = int((y == 0).sum())
    c1 = int((y == 1).sum())
    if min(c0, c1) < min_child:
        return work, {
            **info,
            "reason": "split_creates_too_small_family",
            "counts": [c0, c1],
            "min_child": min_child,
        }

    new_fam_id = int(work["family_id"].max()) + 1

    out = work.copy()
    idx = fam.index.to_numpy()
    out.loc[idx[y == 0], "family_id"] = int(fam_id)
    out.loc[idx[y == 1], "family_id"] = int(new_fam_id)

    return out, {
        **info,
        "mode": "split_by_tone_gmm2",
        "changed": True,
        "new_family_id": int(new_fam_id),
        "counts": [c0, c1],
        "min_child": min_child,
        "refine_features": list(feats),
    }


def _refine_overwide_families(
    work: pd.DataFrame,
    *,
    cfg: HierClusterConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Iterate refinement passes.

    IMPORTANT FIX:
      - do NOT stop when the single worst family cannot be split
      - instead: try the next worst (up to cfg.refine_try_topk_worst_families)
      - stop only if none of the top-K worst families can be split in that pass.
    """
    reports: List[dict] = []
    out = work.copy()

    topk = int(cfg.refine_try_topk_worst_families)
    topk = max(1, topk)

    for _pass in range(int(cfg.max_family_refine_passes)):
        fam_ids = sorted(out["family_id"].unique().tolist())
        if not fam_ids:
            break

        stats: List[Tuple[int, float]] = []
        for fid in fam_ids:
            p90 = _family_de_p90_to_proto(out, int(fid))
            if np.isfinite(p90):
                stats.append((int(fid), float(p90)))

        if not stats:
            break

        stats.sort(key=lambda t: t[1], reverse=True)

        worst_p90 = float(stats[0][1])
        rep_pass: dict = {
            "pass": int(_pass),
            "threshold": float(cfg.family_de_p90_max),
            "worst_family_id": int(stats[0][0]),
            "worst_family_de_p90": worst_p90,
            "changed": False,
            "attempted": [],
            "skipped": [],
        }

        if worst_p90 <= float(cfg.family_de_p90_max):
            reports.append({**rep_pass, "mode": "stop_all_families_within_threshold"})
            break

        chosen: Optional[Tuple[int, float]] = None
        chosen_split: Optional[dict] = None
        out2: Optional[pd.DataFrame] = None

        for fid, p90 in stats[:topk]:
            if float(p90) <= float(cfg.family_de_p90_max):
                break

            out_candidate, rep_split = _split_family_once_by_tone(out, fam_id=int(fid), cfg=cfg)
            rep_pass["attempted"].append({"family_id": int(fid), "de_p90": float(p90), "split": rep_split})

            if rep_split.get("changed", False):
                chosen = (int(fid), float(p90))
                chosen_split = rep_split
                out2 = out_candidate
                break
            else:
                rep_pass["skipped"].append(
                    {"family_id": int(fid), "de_p90": float(p90), "reason": rep_split.get("reason")}
                )

        if out2 is None or chosen is None or chosen_split is None:
            reports.append({**rep_pass, "mode": "stop_no_splittable_family_in_topk"})
            break

        out = out2
        out["family_id"] = _compact_ids(out["family_id"])

        merge_reps: List[dict] = []
        for _m in range(int(cfg.max_family_merge_passes)):
            out, mrep = _merge_small_families_once(
                out,
                family_features=cfg.family_features,
                family_feature_weights=cfg.family_feature_weights,
                min_family_n=int(cfg.min_family_samples_after_fit),
            )
            merge_reps.append(mrep)
            out["family_id"] = _compact_ids(out["family_id"])
            if not mrep.get("changed", False):
                break

        reports.append(
            {
                **rep_pass,
                "changed": True,
                "mode": "split_then_merge",
                "chosen_family_id": int(chosen[0]),
                "chosen_family_de_p90": float(chosen[1]),
                "chosen_split": chosen_split,
                "merge_after_split": merge_reps,
            }
        )

    final_stats = {
        str(int(fid)): float(_family_de_p90_to_proto(out, int(fid)))
        for fid in sorted(out["family_id"].unique().tolist())
    }

    report = {
        "passes": reports,
        "final_family_de_p90": final_stats,
        "family_de_p90_max": float(cfg.family_de_p90_max),
        "family_refine_features": list(cfg.family_refine_features),
        "max_family_refine_passes": int(cfg.max_family_refine_passes),
        "min_family_child_samples_for_refine": int(cfg.min_family_child_samples_for_refine),
        "refine_try_topk_worst_families": int(cfg.refine_try_topk_worst_families),
    }
    return out, report


# ---------------------------------------------------------------------
# Post-fit subcluster cleanup (merge tiny subclusters WITHIN family)
# ---------------------------------------------------------------------

def _merge_small_subclusters_once_in_family(
    fam_df: pd.DataFrame,
    *,
    sub_features: Tuple[str, ...],
    min_subcluster_n: int,
) -> Tuple[pd.DataFrame, dict]:
    counts = fam_df["subcluster_id"].value_counts()
    small = counts[counts < int(min_subcluster_n)].index.tolist()
    big = counts[counts >= int(min_subcluster_n)].index.tolist()

    info = {
        "min_subcluster_n": int(min_subcluster_n),
        "n_subclusters_before": int(counts.size),
        "small_subclusters": [int(x) for x in small],
        "big_subclusters": [int(x) for x in big],
        "small_total_n": int(counts.loc[small].sum()) if small else 0,
    }

    if (not small) or (not big):
        out = fam_df.copy()
        out["subcluster_id"] = _compact_ids(out["subcluster_id"])
        return out, {**info, "changed": False, "mode": "no_merge"}

    prot = (
        fam_df[fam_df["subcluster_id"].isin(big)]
        .groupby("subcluster_id", as_index=False)[list(sub_features)]
        .median()
        .sort_values("subcluster_id")
        .reset_index(drop=True)
    )

    sc = StandardScaler()
    Xs = sc.fit_transform(fam_df[list(sub_features)].to_numpy(float))
    Ps = sc.transform(prot[list(sub_features)].to_numpy(float))
    sc_ids = prot["subcluster_id"].to_numpy(int)

    d2 = ((Xs[:, None, :] - Ps[None, :, :]) ** 2).sum(axis=2)
    nearest_big = sc_ids[np.argmin(d2, axis=1)]

    out = fam_df.copy()
    mask_small = out["subcluster_id"].isin(small).to_numpy()
    out.loc[mask_small, "subcluster_id"] = nearest_big[mask_small]
    out["subcluster_id"] = _compact_ids(out["subcluster_id"])

    counts_after = out["subcluster_id"].value_counts()
    return out, {
        **info,
        "changed": True,
        "mode": "merged_to_big_subclusters",
        "n_subclusters_after": int(counts_after.size),
        "min_subcluster_n_after": int(counts_after.min()),
    }


def _enforce_min_subcluster_size(
    work: pd.DataFrame,
    *,
    sub_features: Tuple[str, ...],
    min_subcluster_n: int,
    max_passes: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    per_family_reports: Dict[str, List[dict]] = {}
    out = work.copy()

    for fam in sorted(out["family_id"].unique().tolist()):
        fam_df = out[out["family_id"] == fam].copy()
        fam_reports: List[dict] = []

        for _ in range(int(max_passes)):
            fam_df, rep = _merge_small_subclusters_once_in_family(
                fam_df,
                sub_features=sub_features,
                min_subcluster_n=int(min_subcluster_n),
            )
            fam_reports.append(rep)

            vc = fam_df["subcluster_id"].value_counts()
            if vc.empty:
                break
            if int(vc.min()) >= int(min_subcluster_n):
                break
            if fam_df["subcluster_id"].nunique() <= 1:
                break

        per_family_reports[str(int(fam))] = fam_reports
        out.loc[fam_df.index, "subcluster_id"] = fam_df["subcluster_id"].astype(int)

    fixups: List[dict] = []
    for fam in sorted(out["family_id"].unique().tolist()):
        fam_rows = out[out["family_id"] == fam]
        if fam_rows.empty:
            continue
        min_n = int(fam_rows["subcluster_id"].value_counts().min())
        if min_n < int(min_subcluster_n):
            out.loc[fam_rows.index, "subcluster_id"] = 0
            fixups.append({"family_id": int(fam), "action": "collapse_to_one_subcluster", "min_n_before": min_n})

    report = {"per_family_merge_passes": per_family_reports, "final_fixups": fixups}
    return out, report


# ---------------------------------------------------------------------
# Labeling (optional, deterministic, explainable)
# ---------------------------------------------------------------------

def _bucket_from_quantiles(x: float, q30: float, q70: float, lo: str, mid: str, hi: str) -> str:
    if x <= q30:
        return lo
    if x >= q70:
        return hi
    return mid


# ---------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------

def _circ_range_deg(h: np.ndarray) -> float:
    """Circular range in degrees (0..360), robust to wrap-around."""
    if h.size == 0:
        return 0.0
    hh = np.sort((h % 360.0).astype(np.float64))
    diffs = np.diff(hh)
    wrap = (hh[0] + 360.0) - hh[-1]
    gaps = np.concatenate([diffs, np.array([wrap])])
    max_gap = float(np.max(gaps))
    return float(360.0 - max_gap)


def _cluster_de_p90_to_proto(df: pd.DataFrame) -> float:
    """dE76 p90 to cluster median Lab prototype."""
    if df.empty:
        return float("nan")
    L0 = float(df["L_lab"].median())
    a0 = float(df["a_lab"].median())
    b0 = float(df["b_lab"].median())
    d = _de76(
        df["L_lab"].to_numpy(float),
        df["a_lab"].to_numpy(float),
        df["b_lab"].to_numpy(float),
        L0, a0, b0,
    )
    return float(np.quantile(d, 0.9))


def _split_cluster_gmm2(
    cdf: pd.DataFrame,
    *,
    features: Tuple[str, ...],
    random_state: int,
    max_iter: int,
    n_init: int,
) -> np.ndarray:
    """Return labels (0/1) from GMM(k=2) on standardized features."""
    X = cdf[list(features)].to_numpy(float)
    Xs = StandardScaler().fit_transform(X)
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=int(random_state),
        max_iter=int(max_iter),
        n_init=int(n_init),
        reg_covar=1e-6,
    )
    gmm.fit(Xs)
    return gmm.predict(Xs).astype(int)


def _repair_overwide_clusters(
    work: pd.DataFrame,
    *,
    cfg: HierClusterConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Post-pass repair:
      - compute per-cluster dE_p90
      - if dE_p90 > cfg.cluster_de_p90_max => attempt split with GMM2
        - if hue span is large => split using (cos_H, sin_H, L, C)
        - else => split using (L, a, b, C)
      - assign the split as a NEW subcluster_id within the same family
      - do NOT change family_id

    IMPORTANT FIXES:
      - enforce global contract: repair cannot create children < cfg.min_samples_per_subcluster
        => min_child_effective = max(cfg.min_cluster_child_samples_for_repair, cfg.min_samples_per_subcluster)
      - try top-K worst clusters per pass (cfg.repair_try_topk_worst_clusters) to avoid looping on an unsplittable worst cluster
      - after each applied split: compact subcluster ids within family + recompute cluster_id
      - stop if no cluster among top-K is splittable in a pass
    """
    out = work.copy()

    report_passes: List[dict] = []
    max_splits = int(cfg.max_cluster_repair_splits)

    min_child_effective = int(max(int(cfg.min_cluster_child_samples_for_repair), int(cfg.min_samples_per_subcluster)))
    topk = int(getattr(cfg, "repair_try_topk_worst_clusters", 8))
    topk = max(1, topk)

    for _pass in range(max_splits):
        # stats per cluster
        stats: List[Tuple[int, float]] = []
        for cid in sorted(out["cluster_id"].unique().tolist()):
            cdf = out[out["cluster_id"] == cid]
            p90 = _cluster_de_p90_to_proto(cdf)
            if np.isfinite(p90):
                stats.append((int(cid), float(p90)))

        if not stats:
            report_passes.append({"pass": int(_pass), "mode": "stop_no_clusters"})
            break

        stats.sort(key=lambda t: t[1], reverse=True)
        worst_cid, worst_p90 = stats[0]

        rep_pass = {
            "pass": int(_pass),
            "threshold": float(cfg.cluster_de_p90_max),
            "worst_cluster_id": int(worst_cid),
            "worst_cluster_de_p90": float(worst_p90),
            "min_child_enforced": int(min_child_effective),
            "topk": int(topk),
            "changed": False,
            "attempted": [],
            "skipped": [],
        }

        if float(worst_p90) <= float(cfg.cluster_de_p90_max):
            report_passes.append({**rep_pass, "mode": "stop_all_clusters_within_threshold"})
            break

        applied_this_pass = False

        for cid, p90 in stats[:topk]:
            if float(p90) <= float(cfg.cluster_de_p90_max):
                break

            cdf = out[out["cluster_id"] == int(cid)].copy()
            n = int(len(cdf))

            attempt_base = {"cluster_id": int(cid), "de_p90": float(p90), "n": n}

            if n < 2 * min_child_effective:
                rep_pass["skipped"].append({**attempt_base, "reason": "not_enough_samples"})
                continue

            hue_span = _circ_range_deg(cdf["H_lab_deg"].to_numpy(float))

            if hue_span >= float(cfg.cluster_hue_span_deg_for_hue_split):
                feats = ("cos_H", "sin_H", "L_lab", "C_lab")
                split_mode = "split_hue_aware"
            else:
                feats = ("L_lab", "a_lab", "b_lab", "C_lab")
                split_mode = "split_tone_aware"

            missing = [c for c in feats if c not in cdf.columns]
            if missing:
                rep_pass["skipped"].append({**attempt_base, "reason": "missing_features", "missing": missing})
                continue

            y = _split_cluster_gmm2(
                cdf,
                features=feats,
                random_state=cfg.random_state,
                max_iter=cfg.max_iter,
                n_init=cfg.n_init,
            )
            c0 = int((y == 0).sum())
            c1 = int((y == 1).sum())

            rep_pass["attempted"].append(
                {
                    **attempt_base,
                    "hue_span_deg": float(hue_span),
                    "counts": [c0, c1],
                    "split_mode": split_mode,
                    "features": list(feats),
                }
            )

            if min(c0, c1) < min_child_effective:
                rep_pass["skipped"].append(
                    {**attempt_base, "reason": "too_small_child", "counts": [c0, c1], "min_child": int(min_child_effective)}
                )
                continue

            # APPLY split
            fam_id = int(cdf["family_id"].iloc[0])
            current_max_sub = int(out.loc[out["family_id"] == fam_id, "subcluster_id"].max())
            new_sub = current_max_sub + 1

            idx = cdf.index.to_numpy()
            base_sub = int(cdf["subcluster_id"].iloc[0])

            out.loc[idx[y == 0], "subcluster_id"] = base_sub
            out.loc[idx[y == 1], "subcluster_id"] = int(new_sub)

            out = _compact_subcluster_ids_within_family(out)
            out["cluster_id"] = (out["family_id"].astype(int) * 100 + out["subcluster_id"].astype(int)).astype(int)

            report_passes.append(
                {
                    **rep_pass,
                    "mode": "split_applied",
                    "changed": True,
                    "chosen_cluster_id": int(cid),
                    "chosen_cluster_de_p90": float(p90),
                    "chosen_counts": [c0, c1],
                    "chosen_split_mode": split_mode,
                    "family_id": int(fam_id),
                    "new_subcluster_id_raw": int(new_sub),
                }
            )
            applied_this_pass = True
            break

        if not applied_this_pass:
            report_passes.append({**rep_pass, "mode": "stop_no_splittable_cluster_in_topk"})
            break

    # snapshot final stats
    final_stats = {}
    for cid in sorted(out["cluster_id"].unique().tolist()):
        cdf = out[out["cluster_id"] == cid]
        final_stats[str(int(cid))] = float(_cluster_de_p90_to_proto(cdf))

    report = {
        "passes": report_passes,
        "final_cluster_de_p90": final_stats,
        "cluster_de_p90_max": float(cfg.cluster_de_p90_max),
        "cluster_hue_span_deg_for_hue_split": float(cfg.cluster_hue_span_deg_for_hue_split),
        "min_cluster_child_samples_for_repair_effective": int(min_child_effective),
        "max_cluster_repair_splits": int(cfg.max_cluster_repair_splits),
        "repair_try_topk_worst_clusters": int(topk),
    }
    return out, report


def fit_hierarchical_color_clusters(
    df: pd.DataFrame,
    *,
    cfg: HierClusterConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    df = _ensure_family_basis(df)

    required = {"L_lab", "a_lab", "b_lab", "C_lab", "H_lab_deg", "cos_H", "sin_H"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Input CSV missing required columns {missing}. "
            "Run enrich_dataset.py first and point to the enriched CSV."
        )

    work = df.dropna(subset=list(required)).copy()
    if work.empty:
        raise ValueError("No rows available after dropping missing Lab/LCH features.")

    if len(cfg.family_feature_weights) != len(cfg.family_features):
        raise ValueError(
            f"family_feature_weights length ({len(cfg.family_feature_weights)}) "
            f"must match family_features length ({len(cfg.family_features)})."
        )

    # ---- Level 1: family via GMM + BIC on (cosH,sinH,C,L)
    X1 = _matrix(work, cfg.family_features)
    X1s = StandardScaler().fit_transform(X1)
    X1s = _apply_feature_weights(X1s, cfg.family_feature_weights)

    fam_gmm, fam_bic_rows = _fit_gmm_bic(
        X1s,
        k_min=cfg.family_k_min,
        k_max=cfg.family_k_max,
        random_state=cfg.random_state,
        max_iter=cfg.max_iter,
        n_init=cfg.n_init,
    )
    work["family_id"] = fam_gmm.predict(X1s).astype(int)

    # ---- Post-fit: merge tiny families to nearest family prototype
    family_merge_report: List[dict] = []
    for _ in range(int(cfg.max_family_merge_passes)):
        work, rep = _merge_small_families_once(
            work,
            family_features=cfg.family_features,
            family_feature_weights=cfg.family_feature_weights,
            min_family_n=int(cfg.min_family_samples_after_fit),
        )
        family_merge_report.append(rep)
        work["family_id"] = _compact_ids(work["family_id"])
        if not rep.get("changed", False):
            break

    # ---- Refine overly broad families (tone split by dE_p90)
    work, family_refine_report = _refine_overwide_families(work, cfg=cfg)

    # ---- Level 2: subcluster per family via GMM + BIC on (L,a,b,C)
    work["subcluster_id"] = 0
    sub_bic_by_family: Dict[str, List[Dict[str, float]]] = {}

    for fam in sorted(work["family_id"].unique().tolist()):
        sub = work[work["family_id"] == fam].copy()
        n = int(len(sub))

        if n < int(cfg.min_samples_per_family_for_split):
            work.loc[sub.index, "subcluster_id"] = 0
            sub_bic_by_family[str(fam)] = [{"k": 1.0, "bic": float("nan"), "aic": float("nan")}]
            continue

        if n < int(cfg.min_samples_per_subcluster * 2):
            work.loc[sub.index, "subcluster_id"] = 0
            sub_bic_by_family[str(fam)] = [{"k": 1.0, "bic": float("nan"), "aic": float("nan")}]
            continue

        X2 = _matrix(sub, cfg.sub_features)
        X2s = StandardScaler().fit_transform(X2)

        max_k_by_n = int(n // int(cfg.min_samples_per_subcluster))
        k_max = int(min(cfg.sub_k_max, max_k_by_n))
        if k_max < 2:
            work.loc[sub.index, "subcluster_id"] = 0
            sub_bic_by_family[str(fam)] = [{"k": 1.0, "bic": float("nan"), "aic": float("nan")}]
            continue

        sub_gmm, sub_bic_rows = _fit_gmm_bic(
            X2s,
            k_min=2,
            k_max=k_max,
            random_state=cfg.random_state,
            max_iter=cfg.max_iter,
            n_init=cfg.n_init,
        )
        sub_bic_by_family[str(fam)] = sub_bic_rows
        work.loc[sub.index, "subcluster_id"] = sub_gmm.predict(X2s).astype(int)

    # ---- Enforce min subcluster size (merge tiny subclusters within each family)
    work, subcluster_merge_report = _enforce_min_subcluster_size(
        work,
        sub_features=cfg.sub_features,
        min_subcluster_n=int(cfg.min_samples_per_subcluster),
        max_passes=int(cfg.max_subcluster_merge_passes),
    )

    # ---- Global cluster_id: stable, unique, deterministic
    work = _compact_subcluster_ids_within_family(work)
    work["cluster_id"] = (work["family_id"].astype(int) * 100 + work["subcluster_id"].astype(int)).astype(int)

    # ---- Optional post-pass repair (must keep contract)
    cluster_repair_report: Dict[str, object] = {"mode": "disabled"}
    if bool(getattr(cfg, "enable_cluster_repair", True)):
        work, cluster_repair_report = _repair_overwide_clusters(work, cfg=cfg)

        # Hard enforcement again (repair can create new subcluster ids; contract must still hold)
        work, subcluster_merge_report_after_repair = _enforce_min_subcluster_size(
            work,
            sub_features=cfg.sub_features,
            min_subcluster_n=int(cfg.min_samples_per_subcluster),
            max_passes=int(cfg.max_subcluster_merge_passes),
        )
        work = _compact_subcluster_ids_within_family(work)
        work["cluster_id"] = (work["family_id"].astype(int) * 100 + work["subcluster_id"].astype(int)).astype(int)

        cluster_repair_report = {
            **cluster_repair_report,
            "post_repair_enforce_min_subcluster": subcluster_merge_report_after_repair,
        }

    # ---- Prototypes
    proto = (
        work.groupby(["cluster_id", "family_id", "subcluster_id"], as_index=False)
        .agg(
            n=("cluster_id", "size"),
            L_lab=("L_lab", "median"),
            a_lab=("a_lab", "median"),
            b_lab=("b_lab", "median"),
            C_lab=("C_lab", "median"),
            H_lab_deg=("H_lab_deg", "median"),
            hue_hsl_deg=("hue_hsl_deg", "median") if "hue_hsl_deg" in work.columns else ("H_lab_deg", "median"),
            sat_hsl=("sat_hsl", "median") if "sat_hsl" in work.columns else ("C_lab", "median"),
            light_hsl=("light_hsl", "median") if "light_hsl" in work.columns else ("L_lab", "median"),
            depth=("depth", "median") if "depth" in work.columns else ("L_lab", "median"),
            warmth=("warmth", "median") if "warmth" in work.columns else ("b_lab", "median"),
            sat_eff=("sat_eff", "median") if "sat_eff" in work.columns else ("C_lab", "median"),
        )
    )

    # ---- Optional: explainable labels from within-family quantiles
    if cfg.enable_auto_labels:
        proto["name"] = ""
        for fam in sorted(proto["family_id"].unique().tolist()):
            psub = proto[proto["family_id"] == fam].copy()
            if psub.empty:
                continue

            fam_rows = work[work["family_id"] == fam]
            qL30, qL70 = fam_rows["L_lab"].quantile([0.3, 0.7]).tolist()
            qC30, qC70 = fam_rows["C_lab"].quantile([0.3, 0.7]).tolist()

            if "warmth" in fam_rows.columns:
                qw30, qw70 = fam_rows["warmth"].quantile([0.3, 0.7]).tolist()
            else:
                qw30, qw70 = 0.0, 0.0

            for idx, r in psub.iterrows():
                L_bucket = _bucket_from_quantiles(float(r["L_lab"]), float(qL30), float(qL70), "deep", "mid", "light")
                C_bucket = _bucket_from_quantiles(float(r["C_lab"]), float(qC30), float(qC70), "muted", "balanced", "vivid")
                if "warmth" in fam_rows.columns:
                    warm_bucket = _bucket_from_quantiles(float(r["warmth"]), float(qw30), float(qw70), "cool", "neutral", "warm")
                else:
                    warm_bucket = "neutral"
                proto.loc[idx, "name"] = f"fam{int(fam)} {L_bucket} {warm_bucket} {C_bucket} sc{int(r['subcluster_id'])}"
    else:
        proto["name"] = proto["cluster_id"].map(lambda i: f"cluster_{int(i)}")

    proto = proto.sort_values(["family_id", "H_lab_deg", "L_lab"]).reset_index(drop=True)

    # ---- Assignments
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
        "family_id",
        "subcluster_id",
        "cluster_id",
    ]
    keep_cols = [c for c in keep_cols if c in work.columns]
    assignments = work[keep_cols].copy()

    report: Dict[str, object] = {
        "family_bic_grid": fam_bic_rows,
        "family_selected_k": int(getattr(fam_gmm, "n_components", -1)),
        "family_feature_weights": list(cfg.family_feature_weights),
        "family_merge_report": family_merge_report,
        "family_refine_report": family_refine_report,
        "family_sizes_final": {str(int(k)): int(v) for k, v in work["family_id"].value_counts().to_dict().items()},
        "sub_bic_grid_by_family": sub_bic_by_family,
        "subcluster_post_merge_report": subcluster_merge_report,
        "cluster_counts": (
            work.groupby(["family_id", "subcluster_id"], as_index=False)
            .size()
            .rename(columns={"size": "n"})
            .to_dict(orient="records")
        ),
        "cluster_id_scheme": "cluster_id = family_id * 100 + subcluster_id",
        "cluster_repair_report": cluster_repair_report,
        "notes": {
            "post_repair": (
                "Post-pass: repair overwide clusters by dE_p90 using GMM2 (hue-aware if circular hue span is large, else tone-aware). "
                "Does not change family_id. After repair, min subcluster size contract is enforced again. "
                "Repair tries top-K worst clusters per pass to avoid looping on an unsplittable worst cluster."
            ),
            "level1": (
                "GMM(full cov) on standardized [cosH,sinH,C,L], k chosen by BIC. "
                "Standardized features are weighted (post-std) to emphasize hue."
            ),
            "level1_post": "Merge tiny families to nearest BIG-family prototype using the same standardized+weighted family feature space.",
            "level1_refine": (
                "Refine overly broad families: compute family dE76 p90 to family proto (median Lab); "
                "if above threshold, try splitting by tone (GMM2 on [L,C]) among top-K worst; "
                "skip unsplittable families instead of stopping."
            ),
            "level2": "GMM(full cov) on standardized [L,a,b,C] per family; k chosen by BIC with min-size constraints.",
            "level2_post": (
                "Contract enforcement: merge tiny subclusters (<min_samples_per_subcluster) "
                "to nearest big subcluster prototype WITHIN family, repeat until satisfied."
            ),
            "family_features": list(cfg.family_features),
            "family_refine_features": list(cfg.family_refine_features),
            "sub_features": list(cfg.sub_features),
            "family_merge": {
                "min_family_samples_after_fit": int(cfg.min_family_samples_after_fit),
                "max_passes": int(cfg.max_family_merge_passes),
            },
            "family_refine": {
                "family_de_p90_max": float(cfg.family_de_p90_max),
                "max_family_refine_passes": int(cfg.max_family_refine_passes),
                "min_family_child_samples_for_refine": int(cfg.min_family_child_samples_for_refine),
                "refine_try_topk_worst_families": int(cfg.refine_try_topk_worst_families),
            },
            "subcluster": {
                "sub_k_max": int(cfg.sub_k_max),
                "min_samples_per_subcluster": int(cfg.min_samples_per_subcluster),
                "max_subcluster_merge_passes": int(cfg.max_subcluster_merge_passes),
            },
            "cluster_repair": {
                "cluster_de_p90_max": float(cfg.cluster_de_p90_max),
                "cluster_hue_span_deg_for_hue_split": float(cfg.cluster_hue_span_deg_for_hue_split),
                "min_cluster_child_samples_for_repair": int(cfg.min_cluster_child_samples_for_repair),
                "min_cluster_child_samples_effective": int(max(cfg.min_cluster_child_samples_for_repair, cfg.min_samples_per_subcluster)),
                "max_cluster_repair_splits": int(cfg.max_cluster_repair_splits),
                "repair_try_topk_worst_clusters": int(cfg.repair_try_topk_worst_clusters),
            },
        },
    }

    return proto, assignments, report


# ---------------------------------------------------------------------
# Plotting (saved to disk)
# ---------------------------------------------------------------------

def _save_plots(assignments: pd.DataFrame, prototypes: pd.DataFrame, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    pdir = _plots_dir(outdir)
    pdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    fam = assignments["family_id"].to_numpy()
    plt.scatter(assignments["a_lab"], assignments["b_lab"], c=fam, s=8, alpha=0.35)
    plt.scatter(
        prototypes["a_lab"],
        prototypes["b_lab"],
        c=prototypes["family_id"],
        s=80,
        alpha=0.9,
        edgecolors="k",
        linewidths=0.3,
    )
    plt.xlabel("a*")
    plt.ylabel("b*")
    plt.title("Lab a* vs b* (family structure)")
    plt.tight_layout()
    plt.savefig(pdir / "hier_a_vs_b_family.png", dpi=160)
    plt.close()

    plt.figure()
    plt.scatter(assignments["H_lab_deg"], assignments["C_lab"], c=fam, s=8, alpha=0.35)
    plt.scatter(
        prototypes["H_lab_deg"],
        prototypes["C_lab"],
        c=prototypes["family_id"],
        s=80,
        alpha=0.9,
        edgecolors="k",
        linewidths=0.3,
    )
    plt.xlabel("Hue (Lab, deg)")
    plt.ylabel("Chroma (C*)")
    plt.title("Hue vs C* (family structure)")
    plt.tight_layout()
    plt.savefig(pdir / "hier_hue_vs_c_family.png", dpi=160)
    plt.close()


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def run_clustering(infile: Path, outdir: Path, *, cfg: HierClusterConfig, save_plots: bool) -> Tuple[Path, Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(infile)

    prototypes, assignments, report = fit_hierarchical_color_clusters(df, cfg=cfg)

    proto_path = outdir / cfg.prototypes_filename
    asg_path = outdir / cfg.assignments_filename
    rep_path = outdir / cfg.report_filename

    prototypes.to_csv(proto_path, index=False)
    assignments.to_csv(asg_path, index=False)
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if save_plots:
        _save_plots(assignments, prototypes, outdir)

    return proto_path, asg_path, rep_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Hierarchical color clustering: families (GMM+BIC) + family refinement (dE_p90 split) "
            "+ subclusters per family (GMM+BIC) + post-merge min-size enforcement."
        ),
    )
    p.add_argument("--infile", type=str, default=str(_default_infile()))
    p.add_argument("--outdir", type=str, default=str(_default_outdir()))

    p.add_argument("--family-k-min", type=int, default=4)
    p.add_argument("--family-k-max", type=int, default=12)

    p.add_argument("--min-family-samples-after-fit", type=int, default=80)
    p.add_argument("--max-family-merge-passes", type=int, default=5)

    # family refinement
    p.add_argument("--family-de-p90-max", type=float, default=30.0)
    p.add_argument("--max-family-refine-passes", type=int, default=8)
    p.add_argument("--min-family-child-samples-for-refine", type=int, default=40)
    p.add_argument("--refine-try-topk-worst-families", type=int, default=8)

    p.add_argument("--sub-k-max", type=int, default=6)
    p.add_argument("--min-family-samples", type=int, default=60)
    p.add_argument("--min-subcluster-samples", type=int, default=40)
    p.add_argument("--max-subcluster-merge-passes", type=int, default=5)

    # cluster repair
    p.add_argument("--no-cluster-repair", action="store_true")
    p.add_argument("--cluster-de-p90-max", type=float, default=30.0)
    p.add_argument("--cluster-hue-span-deg", type=float, default=60.0)
    # IMPORTANT: default must align with min-subcluster-samples (contract)
    p.add_argument("--min-cluster-child-samples", type=int, default=40)
    p.add_argument("--max-cluster-repair-splits", type=int, default=8)
    p.add_argument("--repair-try-topk-worst-clusters", type=int, default=8)

    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    cfg = HierClusterConfig(
        family_k_min=int(args.family_k_min),
        family_k_max=int(args.family_k_max),
        min_family_samples_after_fit=int(args.min_family_samples_after_fit),
        max_family_merge_passes=int(args.max_family_merge_passes),
        family_de_p90_max=float(args.family_de_p90_max),
        max_family_refine_passes=int(args.max_family_refine_passes),
        min_family_child_samples_for_refine=int(args.min_family_child_samples_for_refine),
        refine_try_topk_worst_families=int(args.refine_try_topk_worst_families),
        sub_k_max=int(args.sub_k_max),
        min_samples_per_family_for_split=int(args.min_family_samples),
        min_samples_per_subcluster=int(args.min_subcluster_samples),
        max_subcluster_merge_passes=int(args.max_subcluster_merge_passes),
        enable_auto_labels=not bool(args.no_labels),
        enable_cluster_repair=not bool(args.no_cluster_repair),
        cluster_de_p90_max=float(args.cluster_de_p90_max),
        cluster_hue_span_deg_for_hue_split=float(args.cluster_hue_span_deg),
        min_cluster_child_samples_for_repair=int(args.min_cluster_child_samples),
        max_cluster_repair_splits=int(args.max_cluster_repair_splits),
        repair_try_topk_worst_clusters=int(args.repair_try_topk_worst_clusters),
    )

    proto_path, asg_path, rep_path = run_clustering(
        infile=Path(args.infile),
        outdir=Path(args.outdir),
        cfg=cfg,
        save_plots=not bool(args.no_plots),
    )

    print(str(proto_path))
    print(str(asg_path))
    print(str(rep_path))


if __name__ == "__main__":
    main()
