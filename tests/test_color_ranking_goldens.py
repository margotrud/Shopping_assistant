from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from Shopping_assistant.color.scoring import score_inventory


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _item_key(df: pd.DataFrame) -> pd.Series:
    """
    Stable item identifier used by goldens.
    """
    if {"product_id", "shade_id"}.issubset(df.columns):
        return df["product_id"].astype(str) + "|" + df["shade_id"].astype(str)
    if "item_id" in df.columns:
        return df["item_id"].astype(str)
    raise ValueError("No stable key columns found in scored output.")


def test_color_ranking_goldens() -> None:
    """
    Golden test for color ranking.

    Contract:
    - Uses FIXED scoring calibration
    - Explicit lambda_preference (default 0.0)
    - Must reproduce exactly the top-K item keys stored in goldens
    """

    goldens_path = PROJECT_ROOT / "tests" / "goldens" / "color_ranking_goldens.json"
    assert goldens_path.exists(), f"Missing goldens file: {goldens_path}"

    goldens: List[Dict[str, Any]] = json.loads(
        goldens_path.read_text(encoding="utf-8")
    )

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
    calibration_path = (
        PROJECT_ROOT
        / "data"
        / "models"
        / "color_scoring_calibration.json"
    )

    for p in (inventory_path, prototypes_path, assignments_path, calibration_path):
        assert p.exists(), f"Missing required file: {p}"

    inventory = pd.read_csv(inventory_path)
    prototypes = pd.read_csv(prototypes_path)

    for g in goldens:
        scored = score_inventory(
            inventory=inventory,
            prototypes=prototypes,
            assignments_path=assignments_path,
            cluster_id=int(g["cluster_id"]),
            constraints=str(g.get("constraints", "") or ""),
            lambda_constraints=float(g.get("lambda_constraints", 1.0)),
            lambda_preference=float(g.get("lambda_preference", 0.0)),
            calibration_path=calibration_path,
        )

        topk = int(g["topk"])
        result = _item_key(scored).head(topk).tolist()
        expected = list(g["topk_keys"])

        assert result == expected, (
            f"scenario_id={g['scenario_id']} mismatch\n"
            f"expected={expected[:5]}...\n"
            f"got={result[:5]}..."
        )
