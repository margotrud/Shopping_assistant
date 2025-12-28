# src/Shopping_assistant/io/assets.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd

from Shopping_assistant.io.data_schema import (
    validate_inventory,
    validate_prototypes,
    validate_assignments,
    validate_calibration,
)


@dataclass(frozen=True)
class AssetBundle:
    inventory: pd.DataFrame
    prototypes: pd.DataFrame
    assignments: pd.DataFrame
    calibration: dict


def _inject_cluster_id_if_missing(inventory: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """
    Does:
        If inventory lacks cluster_id, left-merge it from assignments on the best available key(s).
        Prefers (product_id, shade_id) if present in both; else uses (shade_id) if present in both.

    Raises:
        KeyError if no suitable join keys exist.
        ValueError if merge yields too many missing cluster_id values.
    """
    if "cluster_id" in inventory.columns:
        return inventory

    inv = inventory.copy()

    # pick join keys deterministically
    keys: list[str] = []
    if all(c in inv.columns for c in ("product_id", "shade_id")) and all(c in assignments.columns for c in ("product_id", "shade_id")):
        keys = ["product_id", "shade_id"]
    elif "shade_id" in inv.columns and "shade_id" in assignments.columns:
        keys = ["shade_id"]
    else:
        raise KeyError("Cannot inject cluster_id: no compatible key between inventory and assignments.")

    # normalize dtypes for merge stability
    for k in keys:
        inv[k] = inv[k].astype(str)
    asg = assignments.copy()
    for k in keys:
        asg[k] = asg[k].astype(str)

    if "cluster_id" not in asg.columns:
        raise KeyError("assignments missing required column 'cluster_id'.")

    merged = inv.merge(asg[keys + ["cluster_id"]], on=keys, how="left")

    miss = merged["cluster_id"].isna().mean()
    if miss > 0.05:
        raise ValueError(
            f"cluster_id injection failed: {miss:.1%} rows missing after merge on {keys}. "
            "Fix assignments or rebuild inventory with cluster_id."
        )

    return merged


def load_assets(
    *,
    enriched_csv: Path,
    prototypes_csv: Path,
    assignments_csv: Path,
    calibration_json: Path,
) -> AssetBundle:
    inventory = pd.read_csv(enriched_csv)
    prototypes = pd.read_csv(prototypes_csv)
    assignments = pd.read_csv(assignments_csv)
    calibration = json.loads(calibration_json.read_text(encoding="utf-8"))

    # Validate what we can early
    validate_prototypes(prototypes)
    validate_assignments(assignments)
    validate_calibration(calibration)

    # Ensure inventory has cluster_id before strict validation
    inventory = _inject_cluster_id_if_missing(inventory, assignments)

    validate_inventory(inventory)

    return AssetBundle(
        inventory=inventory,
        prototypes=prototypes,
        assignments=assignments,
        calibration=calibration,
    )
