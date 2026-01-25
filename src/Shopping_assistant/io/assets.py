# src/Shopping_assistant/io/assets.py
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
    inventory: pd.DataFrame
    calibration: dict


def load_assets(
    *,
    enriched_csv: Path,
    calibration_json: Path,
) -> AssetBundle:
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
    Does:
        Load AssetBundle from env paths; else strict-discover files under root (default: ./data).
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
