# src/Shopping_assistant/io/data_schema.py
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class DataSchemaError(ValueError):
    pass


def _require_columns(df: pd.DataFrame, cols: Iterable[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataSchemaError(f"{name}: missing columns {missing}")


def validate_inventory(df: pd.DataFrame) -> None:
    _require_columns(
        df,
        ["shade_id", "product_id", "L_lab", "a_lab", "b_lab"],
        name="inventory",
    )


def validate_calibration(cal: dict) -> None:
    # Must match what scoring.py uses (cluster-free).
    required = ["deltaE_ref", "thresholds", "scale_iqr", "scale_std"]
    missing = [k for k in required if k not in cal]
    if missing:
        raise DataSchemaError(f"calibration: missing keys {missing}")

    try:
        ref = float(cal["deltaE_ref"])
    except Exception as e:
        raise DataSchemaError("calibration: deltaE_ref must be a number") from e
    if not np.isfinite(ref) or ref <= 0:
        raise DataSchemaError(f"calibration: invalid deltaE_ref={cal.get('deltaE_ref')}")

    if not isinstance(cal["thresholds"], dict):
        raise DataSchemaError("calibration: thresholds must be a dict")
    if not isinstance(cal["scale_iqr"], dict):
        raise DataSchemaError("calibration: scale_iqr must be a dict")
    if not isinstance(cal["scale_std"], dict):
        raise DataSchemaError("calibration: scale_std must be a dict")
