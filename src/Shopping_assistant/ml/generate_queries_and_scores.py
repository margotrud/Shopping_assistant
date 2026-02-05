# src/Shopping_assistant/ml/generate_queries_and_scores.py
"""
Synthetic query and score generation.

Generates text queries and corresponding color scores
to support model training, evaluation, and analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Reuse your scoring module (dynamic constraints)
from Shopping_assistant.color.scoring import Constraint, QuerySpec, score_shades

import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_enriched() -> Path:
    # adjust if your project still uses enriched_data; CLI can override anyway
    # prefer clean structure: data/enriched/
    root = _project_root()
    cand = [
        root / "data" / "enriched" / "Sephora_lipsticks_enriched.csv",
        root / "data" / "enriched_data" / "Sephora_lipsticks_raw_items_with_chip_rgb_enriched.csv",
    ]
    for p in cand:
        if p.exists():
            return p
    # fallback: find any "*enriched*.csv"
    for p in list((root / "data" / "enriched").glob("*enriched*.csv")) + list((root / "data" / "enriched_data").glob("*enriched*.csv")):
        return p
    # if nothing found, return first candidate (will error clearly)
    return cand[0]


def _default_prototypes() -> Path:
    root = _project_root()
    cand = [
        root / "data" / "enriched" / "color_prototypes_kmeans.csv",
        root / "data" / "enriched_data" / "color_prototypes_kmeans.csv",
    ]
    for p in cand:
        if p.exists():
            return p
    return cand[0]


def _default_assignments() -> Path:
    root = _project_root()
    cand = [
        root / "data" / "enriched" / "color_cluster_assignments.csv",
        root / "data" / "enriched_data" / "color_cluster_assignments.csv",
    ]
    for p in cand:
        if p.exists():
            return p
    return cand[0]


def _default_scores_dir() -> Path:
    return _project_root() / "data" / "scores"


# ---------------------------------------------------------------------
# Config / Templates
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Template:
    """Does: represent a query template used to synthesize preference text."""

    name: str
    constraints: Tuple[Constraint, ...]


def _templates() -> List[Template]:
    """
    Small grid of constraint templates.
    These are GENERIC and data-driven via quantile thresholds inside scoring.py.
    """
    return [
        Template("base", ()),
        Template("not_too_bright", (Constraint("L_lab", "<=", "high", 0.8),)),
        Template("bright", (Constraint("L_lab", ">=", "high", 0.8),)),
        Template("deep", (Constraint("depth", ">=", "high", 0.8),)),
        Template("muted", (Constraint("C_lab", "<=", "medium", 0.8),)),
        Template("vivid", (Constraint("C_lab", ">=", "high", 0.8),)),
        Template("warm", (Constraint("warmth", ">=", "medium", 0.6),)),
        Template("cool", (Constraint("warmth", "<=", "medium", 0.6),)),
        # combos (good for diversity)
        Template("nude_like", (Constraint("C_lab", "<=", "medium", 0.8), Constraint("L_lab", ">=", "medium", 0.4))),
        Template("deep_muted", (Constraint("depth", ">=", "high", 0.7), Constraint("C_lab", "<=", "medium", 0.6))),
    ]


# ---------------------------------------------------------------------
# Cluster selection
# ---------------------------------------------------------------------

def _select_clusters(prototypes: pd.DataFrame, n: int, seed: int) -> List[int]:
    """
    Pick clusters for query generation.
    Preference:
      - if 'n' column exists: pick most populated clusters
      - else: pick uniformly spaced by hue (H_lab_deg) then sample
    """
    if "cluster_id" not in prototypes.columns:
        raise KeyError("prototypes missing 'cluster_id'")

    p = prototypes.copy()
    if "n" in p.columns:
        p = p.sort_values("n", ascending=False)
        ids = p["cluster_id"].astype(int).tolist()
        return ids[:n]

    # fallback: spread by hue if available
    if "H_lab_deg" in p.columns:
        p = p.sort_values("H_lab_deg")
        ids = p["cluster_id"].astype(int).tolist()
        if len(ids) <= n:
            return ids
        # take evenly spaced then shuffle a bit
        idx = np.linspace(0, len(ids) - 1, num=n, dtype=int).tolist()
        sel = [ids[i] for i in idx]
        rng = np.random.default_rng(seed)
        rng.shuffle(sel)
        return sel

    # last fallback: random sample
    rng = np.random.default_rng(seed)
    ids = p["cluster_id"].astype(int).tolist()
    rng.shuffle(ids)
    return ids[:n]


# ---------------------------------------------------------------------
# Ensure cluster_id in enriched df (merge assignments; no fallback needed here)
# ---------------------------------------------------------------------

def _ensure_cluster_id_merge(df: pd.DataFrame, assignments_path: Path) -> pd.DataFrame:
    if "cluster_id" in df.columns:
        return df

    if not assignments_path.exists():
        raise FileNotFoundError(
            f"Missing assignments CSV: {assignments_path}. "
            "Run cluster_colors.py first or pass the correct --assignments path."
        )

    asg = pd.read_csv(assignments_path)

    key_cols: List[str] = []
    if "product_id" in df.columns and "product_id" in asg.columns:
        key_cols.append("product_id")
    if "shade_id" in df.columns and "shade_id" in asg.columns:
        key_cols.append("shade_id")

    if not key_cols:
        raise KeyError("Cannot merge assignments: missing product_id/shade_id keys.")

    if "cluster_id" not in asg.columns:
        raise KeyError("assignments CSV missing 'cluster_id'")

    asg = asg[key_cols + ["cluster_id"]].drop_duplicates()
    merged = df.merge(asg, on=key_cols, how="left")

    ok_rate = float(merged["cluster_id"].notna().mean())
    if ok_rate < 0.95:
        raise ValueError(
            f"Only {ok_rate:.1%} of rows received cluster_id after merge. "
            "Your keys mismatch (product_id/shade_id types or source files differ)."
        )

    return merged


# ---------------------------------------------------------------------
# Query id + filenames
# ---------------------------------------------------------------------

def _query_id(cluster_id: int, template_name: str) -> str:
    return f"q__c{cluster_id:03d}__{template_name}"


def _outpath(scores_dir: Path, qid: str) -> Path:
    return scores_dir / f"scored_{qid}.csv"


# ---------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------

def generate_scores(
    df_enriched: pd.DataFrame,
    prototypes: pd.DataFrame,
    *,
    cluster_ids: Sequence[int],
    templates: Sequence[Template],
    scores_dir: Path,
    lambda_constraints: float,
    overwrite: bool,
    top_n_save: Optional[int],
) -> List[Path]:
    """Does: generate synthetic queries and compute scores against inventory.
    Returns: DataFrame of generated queries with associated score targets.
    """
    scores_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []

    for cid in cluster_ids:
        for t in templates:
            qid = _query_id(cid, t.name)
            out = _outpath(scores_dir, qid)
            if out.exists() and not overwrite:
                continue

            query = QuerySpec(like_cluster_id=int(cid), constraints=tuple(t.constraints))
            scored = score_shades(df_enriched, prototypes, query, lambda_constraints=float(lambda_constraints))

            # attach query_id for downstream group splits
            scored.insert(0, "query_id", qid)

            # keep file sizes reasonable (optional)
            if top_n_save is not None and top_n_save > 0:
                scored = scored.head(int(top_n_save)).copy()

            scored.to_csv(out, index=False)
            written.append(out)

    return written


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate MANY query-dependent score files (multiple cluster_id x constraint templates) for weak supervision."
    )
    p.add_argument("--enriched", type=str, default=str(_default_enriched()), help="Enriched CSV with Lab features.")
    p.add_argument("--prototypes", type=str, default=str(_default_prototypes()), help="KMeans prototypes CSV.")
    p.add_argument("--assignments", type=str, default=str(_default_assignments()), help="Assignments CSV (to add cluster_id).")
    p.add_argument("--scores-dir", type=str, default=str(_default_scores_dir()), help="Output directory for scored_*.csv.")
    p.add_argument("--n-clusters", type=int, default=15, help="How many clusters to generate queries for.")
    p.add_argument("--seed", type=int, default=7, help="Random seed for cluster selection fallback.")
    p.add_argument("--lambda-constraints", type=float, default=1.0, help="Global multiplier for constraint penalties.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing scored files.")
    p.add_argument("--top-n-save", type=int, default=0, help="If >0, save only top-N rows per query (smaller files).")
    args = p.parse_args()

    enriched_path = Path(args.enriched)
    proto_path = Path(args.prototypes)
    asg_path = Path(args.assignments)
    scores_dir = Path(args.scores_dir)

    df = pd.read_csv(enriched_path)
    prototypes = pd.read_csv(proto_path)

    # Ensure cluster_id exists in df (merge; strict)
    df = _ensure_cluster_id_merge(df, asg_path)

    cluster_ids = _select_clusters(prototypes, n=int(args.n_clusters), seed=int(args.seed))
    templates = _templates()

    top_n_save = int(args.top_n_save)
    if top_n_save <= 0:
        top_n_save = None

    written = generate_scores(
        df,
        prototypes,
        cluster_ids=cluster_ids,
        templates=templates,
        scores_dir=scores_dir,
        lambda_constraints=float(args.lambda_constraints),
        overwrite=bool(args.overwrite),
        top_n_save=top_n_save,
    )

    # minimal stdout summary
    logger.debug("written_files=%d", len(written))
    logger.debug("%s", written[0])


if __name__ == "__main__":
    main()
