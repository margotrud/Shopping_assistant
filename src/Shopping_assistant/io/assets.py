# src/Shopping_assistant/io/assets.py
"""Runtime asset loading and bundling.

Does: Loads versioned JSON assets (lexicon, anchors, scoring configs/calibration, weights) and exposes them as a
single AssetBundle consumed by NLP and recommendation layers.
Public API: AssetBundle, load_default_assets(), and any load_* helpers referenced outside this module.
Inputs: optional explicit paths (env or args); defaults to repo-local data/ paths when unspecified.
Outputs: validated AssetBundle with parsed structures ready for fast repeated inference.
Errors: raises FileNotFoundError/JSONDecodeError for missing/invalid assets; raises ValueError for schema violations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import os

import pandas as pd

from Shopping_assistant.io.data_schema import (
    validate_inventory,
    validate_calibration,
)



@dataclass(frozen=True)
class AssetBundle:
    """Does: container for runtime assets used by NLP + scoring + recommender.
    Contains: inventory DataFrame, lexicon/anchors, calibration dict, optional preference weights.
    Used by: recommend_from_text() and related debug helpers.
    """
    inventory: pd.DataFrame
    calibration: dict


def load_assets(
    *,
    enriched_csv: Path,
    calibration_json: Path,
) -> AssetBundle:
    """Does: load runtime assets (inventory + JSON configs) from a root directory.
    Inputs: paths for inventory/enriched CSV and model JSONs; validates schema where relevant.
    Returns: AssetBundle ready to be passed into recommend_from_text().
    """
    inventory = pd.read_csv(enriched_csv)
    calibration = json.loads(calibration_json.read_text(encoding="utf-8"))

    validate_calibration(calibration)
    validate_inventory(inventory)

    return AssetBundle(
        inventory=inventory,
        calibration=calibration,
    )


# --- Default assets loading (env-first, strict discovery fallback) ---


def _env_path(key: str) -> Path | None:
    v = os.environ.get(key, "").strip()
    return None if not v else Path(v)


def _find_one(root: Path, patterns: list[str], *, label: str) -> Path:
    hits: list[Path] = []
    for pat in patterns:
        hits.extend(sorted(root.glob(pat)))
    hits = [p for p in hits if p.is_file()]

    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        raise FileNotFoundError(
            f"Cannot locate {label} under {root}. "
            f"Tried patterns={patterns}. "
            f"Either set env var SA_{label.upper()}_PATH or place exactly one matching file."
        )
    raise FileExistsError(
        f"Ambiguous {label} under {root}: {hits}. "
        f"Keep exactly one matching file or set env var SA_{label.upper()}_PATH."
    )


@lru_cache(maxsize=1)
def load_default_assets(*, root: Path | None = None) -> AssetBundle:
    """
    Load the default AssetBundle used by the recommendation pipeline.

    Does: load inventory CSV and calibration JSON from explicit env paths
    (SA_ENRICHED_CSV_PATH, SA_CALIBRATION_JSON_PATH) if set; otherwise
    discover assets under the given root (default: ./data).
    Returns: AssetBundle with inventory (DataFrame) and calibration config (dict).
    """

    enriched = _env_path("SA_ENRICHED_CSV_PATH")
    calibration = _env_path("SA_CALIBRATION_JSON_PATH")

    if all([enriched, calibration]):
        return load_assets(
            enriched_csv=enriched,
            calibration_json=calibration,
        )

    if root is None:
        root = Path(os.environ.get("SA_ASSETS_ROOT", "data")).resolve()

    enriched = enriched or _find_one(root, ["*enriched*.csv", "*inventory*.csv", "*enriched.csv"], label="enriched_csv")
    calibration = calibration or _find_one(root, ["*calibration*.json"], label="calibration_json")

    return load_assets(
        enriched_csv=enriched,
        calibration_json=calibration,
    )
