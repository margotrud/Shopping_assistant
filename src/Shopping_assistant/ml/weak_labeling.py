# src/Shopping_assistant/ml/weak_labeling.py
from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_scores_dir() -> Path:
    return _project_root() / "data" / "scores"


def _default_enriched_csv() -> Path:
    return _project_root() / "data" / "enriched" / "Sephora_lipsticks_enriched.csv"


def _default_outdir() -> Path:
    return _project_root() / "data" / "ml"


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class WeakLabelConfig:
    top_q: float = 0.15      # top 15% -> positive
    bottom_q: float = 0.15   # bottom 15% -> negative
    seed: int = 7

    # Features for first ML baseline (colors-only)
    feature_cols: Tuple[str, ...] = (
        "L_lab",
        "a_lab",
        "b_lab",
        "C_lab",
        "H_lab_deg",
        "depth",
        "warmth",
        "sat_eff",
    )


# ---------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------

def _load_scores(scores_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(scores_dir / "scored_cluster_*.csv")))
    if not files:
        raise FileNotFoundError(f"No scoring files found in {scores_dir}. Run colors.scoring first.")

    dfs = []
    for fp in files:
        s = pd.read_csv(fp)
        # Require essentials from scoring outputs
        required = {"product_id", "shade_id", "score"}
        missing = required - set(s.columns)
        if missing:
            raise KeyError(f"{fp} missing required columns {sorted(missing)}")
        s["__source_file__"] = Path(fp).name
        dfs.append(s)

    out = pd.concat(dfs, ignore_index=True)
    return out


def _derive_query_id(source_file: str) -> str:
    # For now, use file name as query identifier (deterministic and explicit).
    return source_file.replace(".csv", "")


def build_weak_labels(
    scores: pd.DataFrame,
    *,
    cfg: WeakLabelConfig,
) -> pd.DataFrame:
    """
    Convert scoring outputs to weak labels per query:
      - top_q by score -> y=1
      - bottom_q by score -> y=0
      - middle -> dropped
    """
    scores = scores.copy()
    scores["query_id"] = scores["__source_file__"].map(_derive_query_id)

    rows = []
    for qid, g in scores.groupby("query_id", sort=True):
        g = g.dropna(subset=["score"]).copy()
        if len(g) < 50:
            # too small to label reliably
            continue

        g = g.sort_values("score", ascending=False).reset_index(drop=True)

        n = len(g)
        n_pos = max(1, int(round(cfg.top_q * n)))
        n_neg = max(1, int(round(cfg.bottom_q * n)))

        pos = g.head(n_pos).copy()
        pos["label"] = 1

        neg = g.tail(n_neg).copy()
        neg["label"] = 0

        labeled = pd.concat([pos, neg], ignore_index=True)

        # Keep minimal keys for join + analysis
        labeled = labeled[[
            "query_id",
            "product_id",
            "shade_id",
            "score",
            "label",
        ]]

        rows.append(labeled)

    if not rows:
        raise ValueError("No queries produced labels. Check score files size / top_q / bottom_q.")

    out = pd.concat(rows, ignore_index=True)
    # Remove duplicates if the same (query, shade) appears multiple times
    out = out.drop_duplicates(subset=["query_id", "product_id", "shade_id"], keep="first").reset_index(drop=True)
    return out


def attach_features(
    weak_labels: pd.DataFrame,
    enriched: pd.DataFrame,
    *,
    cfg: WeakLabelConfig,
) -> pd.DataFrame:
    _need = {"product_id", "shade_id"}
    if not _need.issubset(enriched.columns):
        raise KeyError(f"Enriched CSV missing keys: {sorted(_need - set(enriched.columns))}")

    # Ensure features exist
    missing_feats = [c for c in cfg.feature_cols if c not in enriched.columns]
    if missing_feats:
        raise KeyError(f"Enriched CSV missing feature columns: {missing_feats}")

    # Join
    feats = enriched[["product_id", "shade_id", *cfg.feature_cols]].copy()
    feats["product_id"] = feats["product_id"].astype(str)
    feats["shade_id"] = feats["shade_id"].astype(str)

    wl = weak_labels.copy()
    wl["product_id"] = wl["product_id"].astype(str)
    wl["shade_id"] = wl["shade_id"].astype(str)

    merged = wl.merge(feats, on=["product_id", "shade_id"], how="left")

    # Drop rows without features
    merged = merged.dropna(subset=list(cfg.feature_cols)).reset_index(drop=True)
    return merged


def split_and_save(df: pd.DataFrame, outdir: Path, *, cfg: WeakLabelConfig) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Stratify by label; keep query_id mixed (baseline)
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    train_idx, temp_idx = train_test_split(
        df.index.to_numpy(),
        test_size=0.30,
        random_state=cfg.seed,
        stratify=y,
    )
    temp = df.loc[temp_idx]
    y_temp = temp["label"].astype(int)

    valid_idx, test_idx = train_test_split(
        temp.index.to_numpy(),
        test_size=0.50,
        random_state=cfg.seed,
        stratify=y_temp,
    )

    train = df.loc[train_idx].reset_index(drop=True)
    valid = df.loc[valid_idx].reset_index(drop=True)
    test = df.loc[test_idx].reset_index(drop=True)

    df.to_csv(outdir / "weak_labels.csv", index=False)
    train.to_csv(outdir / "train.csv", index=False)
    valid.to_csv(outdir / "valid.csv", index=False)
    test.to_csv(outdir / "test.csv", index=False)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Generate weak labels from scoring outputs and join enriched colors features.")
    p.add_argument("--scores-dir", type=str, default=str(_default_scores_dir()), help="Directory containing scored_cluster_*.csv files.")
    p.add_argument("--enriched", type=str, default=str(_default_enriched_csv()), help="Enriched CSV containing colors features.")
    p.add_argument("--outdir", type=str, default=str(_default_outdir()), help="Output directory for weak labels + splits.")
    p.add_argument("--top-q", type=float, default=0.15, help="Top fraction as positive.")
    p.add_argument("--bottom-q", type=float, default=0.15, help="Bottom fraction as negative.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = p.parse_args()

    cfg = WeakLabelConfig(top_q=float(args.top_q), bottom_q=float(args.bottom_q), seed=int(args.seed))

    scores = _load_scores(Path(args.scores_dir))
    weak = build_weak_labels(scores, cfg=cfg)

    enriched = pd.read_csv(args.enriched)
    merged = attach_features(weak, enriched, cfg=cfg)

    split_and_save(merged, Path(args.outdir), cfg=cfg)
    logger.debug("%s", Path(args.outdir).resolve() / "weak_labels.csv")


if __name__ == "__main__":
    main()
