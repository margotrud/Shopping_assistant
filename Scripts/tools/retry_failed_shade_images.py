from __future__ import annotations

import argparse
import mimetypes
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests


MEDIA_VARIANTS = [
    "media_zoom",
    "media_main",
    "media_hero",
    "media_swatch",
    "media_full",
    "media_detail",
    "media_thumbnail",  # fallback volontaire
]

EXT_VARIANTS = ["jpg", "jpeg", "webp", "avif", "png"]


def _strip_query(url: str) -> str:
    return url.split("?", 1)[0]


def _with_query(url: str, w: int = 900, h: int = 900) -> str:
    # garde le pattern sephora.eu, mais ne casse pas si déjà un query
    return f"{url}?scaleWidth={w}&scaleHeight={h}&scaleMode=fit"


def _make_candidates_from_chip(chip_url: str) -> list[str]:
    base = _strip_query(chip_url)

    # attend un pattern "...-media_thumbnail.<ext>"
    m = re.search(r"^(.*)-media_[a-zA-Z0-9]+\.([a-zA-Z0-9]+)$", base)
    if not m:
        # si pattern inattendu, on tente quand même le base lui-même
        return [chip_url, base]

    prefix = m.group(1)

    out: list[str] = []
    for media in MEDIA_VARIANTS:
        for ext in EXT_VARIANTS:
            out.append(_with_query(f"{prefix}-{media}.{ext}", 900, 900))
            # aussi une version sans upscale
            out.append(f"{prefix}-{media}.{ext}")
    # en dernier : chip_url exact (thumbnail avec query)
    out.append(chip_url)
    return out


def _guess_ext_from_content_type(content_type: str) -> Optional[str]:
    # ex: "image/webp"
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    if not ct.startswith("image/"):
        return None
    ext = ct.split("/", 1)[1]
    # normalisations usuelles
    if ext == "jpeg":
        return "jpg"
    return ext


def _download_first_ok(urls: Iterable[str], timeout: float = 20.0) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
    s = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        "Referer": "https://www.sephora.fr/",
    }

    for u in urls:
        try:
            r = s.get(u, headers=headers, timeout=timeout)
            if r.status_code == 200 and r.content:
                ct = r.headers.get("Content-Type", "")
                return r.content, ct, u
        except Exception:
            continue
    return None, None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_size_kb", type=int, default=5)  # filtre anti-faux 200 (tiny)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.errors_csv)

    rows_ok = 0
    rows_fail = 0

    for _, row in df.iterrows():
        product_id = str(row["product_id"])
        shade_id = str(row["shade_id"])
        chip_url = str(row["chip_url"])

        candidates = _make_candidates_from_chip(chip_url)
        blob, content_type, used_url = _download_first_ok(candidates)

        if not blob or (len(blob) < args.min_size_kb * 1024):
            rows_fail += 1
            print(f"[FAIL] {product_id} {shade_id} (no candidate worked)")
            continue

        ext = _guess_ext_from_content_type(content_type or "") or "img"
        out_path = out_dir / f"{product_id}__{shade_id}.{ext}"
        out_path.write_bytes(blob)

        rows_ok += 1
        print(f"[OK]  {product_id} {shade_id} -> {out_path.name} (from {used_url})")

    print(f"\nDone. ok={rows_ok} fail={rows_fail}")


if __name__ == "__main__":
    main()
