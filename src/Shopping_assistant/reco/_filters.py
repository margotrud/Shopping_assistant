# src/Shopping_assistant/reco/_filters.py
"""
Inventory filtering utilities.

Applies validity checks and domain-specific exclusion rules
to product inventories prior to pooling and scoring.
"""

from __future__ import annotations

import pandas as pd

from ._constants import _BAD_PRODUCT_RE


def _filter_invalid_products(inv: pd.DataFrame) -> pd.DataFrame:
    if inv is None or inv.empty:
        return inv
    txt = (inv.get("product_name", "").astype(str) + " " + inv.get("shade_name", "").astype(str)).fillna("")
    keep = ~txt.str.contains(_BAD_PRODUCT_RE, regex=True)
    return inv.loc[keep].copy()
