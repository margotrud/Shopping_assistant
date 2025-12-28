# Scripts/enrich_sephora_chip_colors.py
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_CSV = PROJECT_ROOT / "data" / "Sephora_lipsticks_raw_items.csv"
OUT_CSV = PROJECT_ROOT / "data" / "Sephora_lipsticks_raw_items_with_chip_rgb.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "_chip_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Referer": "https://www.sephora.fr/",
    "Connection": "keep-alive",
}


def _sleep(i: int, base: float = 0.25) -> None:
    time.sleep(base + (i % 3) * 0.1)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _robust_chip_rgb_from_image_bytes(img_bytes: bytes) -> Optional[Tuple[int, int, int]]:
    """
    Robust estimate:
    - convert to RGB
    - crop center region (avoid borders)
    - compute per-channel median
    """
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None

    w, h = im.size
    if w < 10 or h < 10:
        return None

    # central crop: keep middle 50% area
    x0 = int(w * 0.25)
    x1 = int(w * 0.75)
    y0 = int(h * 0.25)
    y1 = int(h * 0.75)
    imc = im.crop((x0, y0, x1, y1))

    # downsample to speed
    imc = imc.resize((32, 32), resample=Image.BILINEAR)

    px = list(imc.getdata())
    if not px:
        return None

    rs = sorted(p[0] for p in px)
    gs = sorted(p[1] for p in px)
    bs = sorted(p[2] for p in px)

    mid = len(px) // 2
    return (int(rs[mid]), int(gs[mid]), int(bs[mid]))


def _cache_key(product_id: str, shade_id: str) -> str:
    return f"{product_id}__{shade_id}.bin"


def _download_with_retries(session: requests.Session, url: str, *, tries: int = 4, timeout: float = 25.0) -> Optional[bytes]:
    for t in range(1, tries + 1):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
            # retry on transient codes
            if r.status_code in (429, 500, 502, 503, 504):
                _sleep(t, base=0.7)
                continue
            return None
        except Exception:
            _sleep(t, base=0.7)
    return None


def main() -> None:
    df = pd.read_csv(IN_CSV)

    required = {"product_id", "shade_id", "chip_url"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {IN_CSV.name}: {sorted(missing)}")

    # prepare output cols
    if "chip_rgb" not in df.columns:
        df["chip_rgb"] = None
    if "chip_hex" not in df.columns:
        df["chip_hex"] = None

    s = requests.Session()
    s.headers.update(HEADERS)

    for i, row in df.iterrows():
        if pd.notna(row.get("chip_hex")) and str(row.get("chip_hex")).strip():
            continue

        product_id = str(row["product_id"])
        shade_id = str(row["shade_id"])
        chip_url = str(row["chip_url"]) if pd.notna(row["chip_url"]) else ""
        if not chip_url:
            continue

        ck = _cache_key(product_id, shade_id)
        cache_path = CACHE_DIR / ck

        if cache_path.exists():
            img_bytes = cache_path.read_bytes()
        else:
            img_bytes = _download_with_retries(s, chip_url)
            if not img_bytes:
                continue
            cache_path.write_bytes(img_bytes)

        rgb = _robust_chip_rgb_from_image_bytes(img_bytes)
        if not rgb:
            continue

        df.at[i, "chip_rgb"] = f"{rgb[0]},{rgb[1]},{rgb[2]}"
        df.at[i, "chip_hex"] = _rgb_to_hex(rgb)

        if i % 50 == 0:
            print(f"[chip] {i+1}/{len(df)}")

        _sleep(i)

    df.to_csv(OUT_CSV, index=False)
    print(f"[out] {OUT_CSV}")


if __name__ == "__main__":
    main()
