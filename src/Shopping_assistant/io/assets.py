# src/Shopping_assistant/io/assets.py
"""Runtime asset loading and bundling.

Does: Loads runtime assets (inventory CSV + calibration JSON) and exposes them as a single AssetBundle.
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

from Shopping_assistant.io.data_schema import validate_calibration, validate_inventory


def _project_root() -> Path:
    # .../src/Shopping_assistant/io/assets.py -> parents[3] = pythonProject/
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class AssetBundle:
    """Does: container for runtime assets used by scoring + recommender.
    Contains: inventory DataFrame, calibration dict.
    Used by: recommend_from_text() and debug helpers.
    """

    inventory: pd.DataFrame
    calibration: dict


def load_assets(*, enriched_csv: Path, calibration_json: Path) -> AssetBundle:
    """Does: load runtime assets from explicit file paths.
    Inputs: enriched/inventory CSV path and calibration JSON path.
    Returns: validated AssetBundle.
    """
    inventory = pd.read_csv(enriched_csv)
    calibration = json.loads(calibration_json.read_text(encoding="utf-8"))

    validate_calibration(calibration)
    validate_inventory(inventory)

    return AssetBundle(inventory=inventory, calibration=calibration)


def _env_path(key: str) -> Path | None:
    v = os.environ.get(key, "").strip()
    return None if not v else Path(v)


def _find_one(root: Path, patterns: list[str], *, label: str) -> Path:
    """Does: locate exactly one file matching patterns under root (recursive, deduped).
    Inputs: root directory and glob patterns (e.g. '*enriched*.csv').
    Returns: unique matching Path.
    """
    hits: list[Path] = []
    for pat in patterns:
        hits.extend(root.rglob(pat))

    # Deduplicate across overlapping patterns
    uniq = sorted({p.resolve() for p in hits if p.is_file()})

    if len(uniq) == 1:
        return uniq[0]
    if len(uniq) == 0:
        raise FileNotFoundError(
            f"Cannot locate {label} under {root}. "
            f"Tried patterns={patterns} (recursive). "
            f"Either set env var SA_{label.upper()}_PATH or place exactly one matching file."
        )
    raise FileExistsError(
        f"Ambiguous {label} under {root}: {uniq}. "
        f"Keep exactly one matching file or set env var SA_{label.upper()}_PATH."
    )


@lru_cache(maxsize=1)
def load_default_assets(*, root: Path | None = None) -> AssetBundle:
    """Does: load inventory CSV + calibration JSON (env-first; otherwise repo-local discovery).
    Inputs: optional root; env overrides SA_ENRICHED_CSV_PATH / SA_CALIBRATION_JSON_PATH / SA_ASSETS_ROOT.
    Returns: AssetBundle with validated inventory + calibration.
    """
    enriched = _env_path("SA_ENRICHED_CSV_PATH")
    calibration = _env_path("SA_CALIBRATION_JSON_PATH")

    if enriched and calibration:
        return load_assets(enriched_csv=enriched, calibration_json=calibration)

    if root is None:
        default_root = _project_root() / "data"
        root = Path(os.environ.get("SA_ASSETS_ROOT", str(default_root))).resolve()

    enriched = enriched or _find_one(root, ["*enriched*.csv", "*inventory*.csv", "*enriched.csv"], label="enriched_csv")
    calibration = calibration or _find_one(root, ["*calibration*.json"], label="calibration_json")

    return load_assets(enriched_csv=enriched, calibration_json=calibration)
