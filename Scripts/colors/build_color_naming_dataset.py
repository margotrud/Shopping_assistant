# Scripts/colors/build_color_naming_dataset.py

from __future__ import annotations

import sys
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
from Shopping_assistant.reco._colorconv import _hex_to_lab

# ------------------------------------------------------------------------------
# Bootstrap project root
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------------------
# Constants (copied & simplified)
# ------------------------------------------------------------------------------
COLOR_NAME_LISTS_BLOB_URLS = [
    "https://unpkg.com/color-name-lists/dist/colorlists.json",
    "https://cdn.jsdelivr.net/npm/color-name-lists/dist/colorlists.json",
    "https://raw.githubusercontent.com/meodai/color-names/master/dist/colorlists.json",
]

OUT_PATH = ROOT / "data" / "colors" / "color_naming_dataset.parquet"
CACHE_DIR = ROOT / "data" / "cache" / "color_naming"

HEX_RE = set("0123456789abcdef")

# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _norm_hex(h: str) -> Optional[str]:
    h = str(h).strip().lower()
    if not h:
        return None
    if not h.startswith("#"):
        h = "#" + h
    if len(h) != 7:
        return None
    if any(c not in HEX_RE for c in h[1:]):
        return None
    return h


def _norm_label(s: str) -> str:
    return str(s).strip().lower()


# ------------------------------------------------------------------------------
# Color conversion (MUST match build_chips_lab.py)
# ------------------------------------------------------------------------------
def hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

# ------------------------------------------------------------------------------
# Fetch + cache color-name-lists blob
# ------------------------------------------------------------------------------
def fetch_blob() -> Dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for url in COLOR_NAME_LISTS_BLOB_URLS:
        cache_path = CACHE_DIR / f"{_sha1(url)}.json"
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(data, f)
            return data
        except Exception:
            continue

    raise RuntimeError("Failed to fetch color-name-lists blob")


# ------------------------------------------------------------------------------
# Build dataset
# ------------------------------------------------------------------------------
def main():
    blob = fetch_blob()

    rows: List[Dict[str, Any]] = []

    lists = blob.get("lists") if isinstance(blob.get("lists"), dict) else blob

    for list_key, items in lists.items():
        if not isinstance(items, list):
            continue

        for it in items:
            if not isinstance(it, dict):
                continue

            name = it.get("name") or it.get("title") or it.get("label")
            hx = it.get("hex") or it.get("value") or it.get("colour")

            if not isinstance(name, str) or not isinstance(hx, str):
                continue

            hx = _norm_hex(hx)
            if not hx:
                continue

            try:
                L, a, b = _hex_to_lab(hx)
            except Exception:
                continue

            rows.append(
                {
                    "L": L,
                    "a": a,
                    "b": b,
                    "label": _norm_label(name),
                    "source": f"colors-name-lists:{list_key}",
                }
            )

    df = pd.DataFrame(rows)
    df = df.dropna().drop_duplicates()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"[OK] wrote {len(df)} rows → {OUT_PATH}")
    print(df["label"].value_counts().head(20))


if __name__ == "__main__":
    main()
