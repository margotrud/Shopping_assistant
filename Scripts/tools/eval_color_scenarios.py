from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from Shopping_assistant.color.scoring import score_inventory


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _find_scenarios_path() -> Path:
    hits = sorted((PROJECT_ROOT / "data").glob("**/color_scenarios.csv"))
    if not hits:
        raise FileNotFoundError("color_scenarios.csv not found under data/")
    return hits[0]


def main() -> None:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    scenarios_path = _find_scenarios_path()
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
    calibration_path = PROJECT_ROOT / "data" / "models" / "color_scoring_calibration.json"
    config_path = PROJECT_ROOT / "data" / "models" / "color_scoring_config.json"

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    scenarios = pd.read_csv(scenarios_path)
    inventory = pd.read_csv(inventory_path)
    prototypes = pd.read_csv(prototypes_path)
    assignments = pd.read_csv(assignments_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # Normalize keys (critical for merges)
    for df in (inventory, assignments):
        df["product_id"] = df["product_id"].astype(str)
        df["shade_id"] = df["shade_id"].astype(str)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    lam_c = float(config["scoring"]["lambda_constraints"])
    lam_p = float(config["scoring"]["lambda_preference"])
    topk_default = int(config["evaluation"].get("topk", 20))

    rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Evaluation loop (scenario-level)
    # ------------------------------------------------------------------
    for _, r in scenarios.iterrows():
        row = r.to_dict()
        cluster_id = int(row["cluster_id"])
        constraints = str(row.get("constraints", "") or "")
        topk = int(row.get("topk", topk_default))

        # structural sanity check
        n_candidates = int((assignments["cluster_id"] == cluster_id).sum())

        # scoring
        scored = score_inventory(
            inventory=inventory,
            prototypes=prototypes,
            assignments_path=assignments_path,
            cluster_id=cluster_id,
            constraints=constraints,
            lambda_constraints=lam_c,
            lambda_preference=lam_p,
            calibration_path=calibration_path,
        ).head(topk)

        # merge back physical color dimensions
        topm = scored.merge(
            inventory,
            on=["product_id", "shade_id"],
            how="left",
            suffixes=("", "__inv"),
        )

        rows.append(
            {
                "cluster_id": cluster_id,
                "constraints": constraints,
                "topk": topk,
                "n_candidates_in_cluster": n_candidates,
                "mean_deltaE_norm": float(scored["deltaE_norm"].mean()),
                "mean_penalty_norm": float(scored["constraint_penalty_norm"].mean()),
                "mean_L_lab_topk": float(topm["L_lab"].mean()) if "L_lab" in topm else np.nan,
                "mean_C_lab_topk": float(topm["C_lab"].mean()) if "C_lab" in topm else np.nan,
                "mean_sat_eff_topk": float(topm["sat_eff"].mean()) if "sat_eff" in topm else np.nan,
            }
        )

    # ------------------------------------------------------------------
    # Write scenario report
    # ------------------------------------------------------------------
    out = pd.DataFrame(rows)
    outdir = PROJECT_ROOT / "data" / "reports"
    outdir.mkdir(parents=True, exist_ok=True)

    out_path = outdir / "eval_color_scenarios.csv"
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")

    # ------------------------------------------------------------------
    # Cluster-level aggregation (fidelity-focused)
    # ------------------------------------------------------------------
    def _p90(x: pd.Series) -> float:
        x = x.dropna().to_numpy(float)
        if x.size == 0:
            return float("nan")
        return float(np.percentile(x, 90))

    cluster_summary = (
        out.groupby("cluster_id", as_index=False)
        .agg(
            n_scenarios=("cluster_id", "size"),
            mean_deltaE_norm_scenarios=("mean_deltaE_norm", "mean"),
            p90_deltaE_norm_scenarios=("mean_deltaE_norm", _p90),
            mean_penalty_norm_scenarios=("mean_penalty_norm", "mean"),
            mean_candidates=("n_candidates_in_cluster", "mean"),
            mean_L_lab_topk=("mean_L_lab_topk", "mean"),
            mean_C_lab_topk=("mean_C_lab_topk", "mean"),
            mean_sat_eff_topk=("mean_sat_eff_topk", "mean"),
        )
        .sort_values(["mean_deltaE_norm_scenarios"], ascending=False)
        .reset_index(drop=True)
    )

    out_clusters_path = outdir / "eval_color_clusters.csv"
    cluster_summary.to_csv(out_clusters_path, index=False)
    print(f"[OK] wrote {out_clusters_path}")

    # ------------------------------------------------------------------
    # All-clusters diagnostics (independent of scenarios.csv)
    # ------------------------------------------------------------------
    # Build cluster list from assignments (source of truth)
    all_clusters = (
        assignments["cluster_id"]
        .dropna()
        .astype(int)
        .value_counts()
        .rename_axis("cluster_id")
        .reset_index(name="n_candidates")
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )

    out_all_clusters_path = outdir / "all_clusters_candidates.csv"
    all_clusters.to_csv(out_all_clusters_path, index=False)
    print(f"[OK] wrote {out_all_clusters_path}")


if __name__ == "__main__":
    main()
