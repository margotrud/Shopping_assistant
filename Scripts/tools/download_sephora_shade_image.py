# Scripts/tools/download_sephora_shade_image.py
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

# =========================
# CLI / CONFIG
# =========================

DEFAULT_CSV = Path(
    os.environ.get(
        "SA_SEPHORA_CSV_PATH",
        "data/Sephora_lipsticks_raw_items_with_chip_rgb.csv",
    )
)

DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "SA_IMAGE_CACHE_DIR",
        "data/_product_image_cache",
    )
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Sephora shade images.")
    p.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to Sephora CSV (default: env SA_SEPHORA_CSV_PATH or data/...csv).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for downloaded images (default: env SA_IMAGE_CACHE_DIR or data/_product_image_cache).",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=int(os.environ.get("SA_IMAGE_SIZE", "900")),
        help="Square size (px) for requested images (default: env SA_IMAGE_SIZE or 900).",
    )
    p.add_argument(
        "--prefer",
        type=str,
        choices=["swatch", "zoom"],
        default=os.environ.get("SA_IMAGE_PREFER", "swatch"),
        help="Prefer swatch or zoom URLs (default: env SA_IMAGE_PREFER or swatch).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: False).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("SA_IMAGE_TIMEOUT", "20")),
        help="Request timeout in seconds (default: env SA_IMAGE_TIMEOUT or 20).",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=int(os.environ.get("SA_IMAGE_RETRIES", "2")),
        help="Retries per candidate URL (default: env SA_IMAGE_RETRIES or 2).",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=float(os.environ.get("SA_IMAGE_SLEEP", "0.20")),
        help="Sleep between rows (seconds) (default: env SA_IMAGE_SLEEP or 0.20).",
    )
    return p.parse_args()


# Accept .jpg and .jpeg thumbnails
_THUMB_RE = re.compile(r"-media_thumbnail\.(?P<ext>jpe?g)$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class ShadeRow:
    product_id: str
    shade_id: str
    brand_name: str
    product_name: str
    shade_name: str
    url: str
    chip_url: str


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ext_from_content_type(ct: str) -> str:
    ct = (ct or "").split(";")[0].strip().lower()
    return {
        "image/avif": ".avif",
        "image/jpeg": ".jpg",  # normalize jpeg -> .jpg locally
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(ct, "")


def _sniff_ext_from_bytes(b: bytes) -> str:
    if len(b) >= 3 and b[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if len(b) >= 8 and b[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return ".webp"
    # ISO BMFF: 'ftyp' at offset 4, brands: avif/avis/heic/heif (common on CDNs)
    if len(b) >= 12 and b[4:8] == b"ftyp" and b[8:12] in {b"avif", b"avis", b"heic", b"heif"}:
        return ".avif"
    return ""


def _build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "image",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "cross-site",
        }
    )
    return s


def _build_center_image_url_candidates_from_chip_url(
    chip_url: str,
    *,
    size: int,
    prefer: str,
) -> list[str]:
    """
    Convert PIM chip thumbnail URL to a list of candidate center-packshot URLs.

    Key fix:
      Some products use -media_swatch.jpeg (NOT .jpg).
      So we try both .jpg and .jpeg for swatch/zoom.

    chip_url example:
      .../<shade_id>-media_thumbnail.jpg?...  or ...thumbnail.jpeg?...

    Returns candidates ordered by:
      1) prefer swatch/zoom
      2) other (zoom/swatch)
      3) extension attempts: original thumb ext first, then the other
    """
    if prefer not in {"swatch", "zoom"}:
        raise ValueError("prefer must be 'swatch' or 'zoom'")

    parts = urlsplit(chip_url)
    path = parts.path

    m = _THUMB_RE.search(path)
    if not m:
        raise ValueError(f"chip_url does not end with -media_thumbnail.(jpg|jpeg): {chip_url}")

    thumb_ext = m.group("ext").lower()  # "jpg" or "jpeg"
    other = "zoom" if prefer == "swatch" else "swatch"

    # Base path WITHOUT extension:
    # ".../367891-media_thumbnail.jpg" -> ".../367891-media_thumbnail"
    base = _THUMB_RE.sub("-media_thumbnail", path)

    # Extension order: try original ext first, then the other
    ext_order = [thumb_ext, "jpeg" if thumb_ext == "jpg" else "jpg"]
    ext_order = list(dict.fromkeys(ext_order))

    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    q.update({"scaleWidth": str(size), "scaleHeight": str(size), "scaleMode": "fit"})
    query = urlencode(q)

    candidates: list[str] = []
    for kind in (prefer, other):
        for ext in ext_order:
            center_path = base.replace("-media_thumbnail", f"-media_{kind}") + f".{ext}"
            candidates.append(urlunsplit((parts.scheme, parts.netloc, center_path, query, "")))

    return candidates


def _download_to_final_path(
    s: requests.Session,
    url: str,
    out_dir: Path,
    base_name: str,
    *,
    referer: Optional[str],
    timeout: int,
    overwrite: bool,
) -> tuple[Optional[Path], str]:
    """
    One GET:
      - stream to temp file
      - decide ext from Content-Type, else sniff magic bytes
      - atomically move to final file name
    Returns (path or None, error_message).
    """
    headers = {}
    if referer:
        headers["Referer"] = referer

    tmp_path = out_dir / f"{base_name}.part"

    try:
        with s.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True) as r:
            if r.status_code >= 400:
                return None, f"HTTPError: GET {r.status_code} url={url}"

            ct = r.headers.get("Content-Type", "")
            ext = _ext_from_content_type(ct)

            head = b""
            with tmp_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    if not chunk:
                        continue
                    if len(head) < 64:
                        head += chunk[: (64 - len(head))]
                    f.write(chunk)

        if not ext:
            ext = _sniff_ext_from_bytes(head)

        if not ext:
            tmp_path.unlink(missing_ok=True)
            return None, f"UnknownImageType: ct={ct!r} url={url}"

        final_path = out_dir / f"{base_name}{ext}"

        if final_path.exists() and not overwrite:
            tmp_path.unlink(missing_ok=True)
            return final_path, ""

        tmp_path.replace(final_path)
        return final_path, ""
    except Exception as e:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None, f"{type(e).__name__}: {e}"


def _download_one(
    s: requests.Session,
    row: ShadeRow,
    out_dir: Path,
    *,
    size: int,
    prefer: str,
    overwrite: bool,
    timeout: int,
    retries: int,
) -> tuple[Optional[Path], str, str]:
    """
    Returns: (image_path or None, center_url_used, error_message)
    - fast reuse if already exists
    - tries swatch/zoom + .jpg/.jpeg candidates
    """
    base_name = f"{row.product_id}__{row.shade_id}"

    # fast reuse if already exists
    if not overwrite:
        for ext in (".jpg", ".png", ".webp", ".avif"):
            p = out_dir / f"{base_name}{ext}"
            if p.exists():
                return p, "", ""

    try:
        candidates = _build_center_image_url_candidates_from_chip_url(
            row.chip_url,
            size=size,
            prefer=prefer,
        )
    except Exception as e:
        return None, "", f"{type(e).__name__}: {e}"

    last_err = ""
    for attempt_url in candidates:
        for attempt in range(max(1, retries + 1)):
            path, err = _download_to_final_path(
                s,
                attempt_url,
                out_dir,
                base_name,
                referer=row.url or "https://www.sephora.fr/",
                timeout=timeout,
                overwrite=overwrite,
            )
            if path is not None:
                return path, attempt_url, ""

            last_err = err or "unknown error"
            time.sleep(0.25 * (attempt + 1))  # basic backoff

    if candidates:
        return None, candidates[0], f"{last_err} (n_candidates={len(candidates)})"
    return None, "", last_err or "unknown error"


def run(
    *,
    csv_path: Path,
    out_dir: Path,
    image_size: int,
    prefer: str,
    overwrite: bool,
    timeout: int,
    retries: int,
    sleep_between: float,
) -> Path:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = _ensure_dir(out_dir)
    index_csv_path = out_dir / "images_index.csv"
    errors_csv_path = out_dir / "images_errors.csv"

    s = _build_session()

    with (
        csv_path.open("r", encoding="utf-8", newline="") as f_in,
        index_csv_path.open("w", encoding="utf-8", newline="") as f_idx,
        errors_csv_path.open("w", encoding="utf-8", newline="") as f_err,
    ):
        reader = csv.DictReader(f_in)
        required = {"product_id", "shade_id", "chip_url"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        fieldnames = [
            "row_idx",
            "product_id",
            "shade_id",
            "shade_name",
            "brand_name",
            "product_name",
            "page_url",
            "chip_url",
            "center_url_used",
            "image_file",
            "status",
            "error",
        ]

        writer = csv.DictWriter(f_idx, fieldnames=fieldnames)
        err_writer = csv.DictWriter(f_err, fieldnames=fieldnames)
        writer.writeheader()
        err_writer.writeheader()

        ok_n = 0
        err_n = 0

        for row_idx, r in enumerate(reader, start=1):
            row = ShadeRow(
                product_id=(r.get("product_id") or "").strip(),
                shade_id=(r.get("shade_id") or "").strip(),
                brand_name=(r.get("brand_name") or "").strip(),
                product_name=(r.get("product_name") or "").strip(),
                shade_name=(r.get("shade_name") or "").strip(),
                url=(r.get("url") or "").strip(),
                chip_url=(r.get("chip_url") or "").strip(),
            )

            if not row.product_id or not row.shade_id or not row.chip_url:
                err_n += 1
                payload = {
                    "row_idx": row_idx,
                    "product_id": row.product_id,
                    "shade_id": row.shade_id,
                    "shade_name": row.shade_name,
                    "brand_name": row.brand_name,
                    "product_name": row.product_name,
                    "page_url": row.url,
                    "chip_url": row.chip_url,
                    "center_url_used": "",
                    "image_file": "",
                    "status": "error",
                    "error": "missing required values (product_id/shade_id/chip_url)",
                }
                writer.writerow(payload)
                err_writer.writerow(payload)
                continue

            image_path, center_url_used, error_msg = _download_one(
                s,
                row,
                out_dir,
                size=image_size,
                prefer=prefer,
                overwrite=overwrite,
                timeout=timeout,
                retries=retries,
            )

            if image_path is not None:
                ok_n += 1
                payload = {
                    "row_idx": row_idx,
                    "product_id": row.product_id,
                    "shade_id": row.shade_id,
                    "shade_name": row.shade_name,
                    "brand_name": row.brand_name,
                    "product_name": row.product_name,
                    "page_url": row.url,
                    "chip_url": row.chip_url,
                    "center_url_used": center_url_used,
                    "image_file": str(image_path.as_posix()),
                    "status": "ok",
                    "error": "",
                }
                writer.writerow(payload)
                print(f"[ok]  {row.product_id} {row.shade_id} -> {image_path.as_posix()}")
            else:
                err_n += 1
                payload = {
                    "row_idx": row_idx,
                    "product_id": row.product_id,
                    "shade_id": row.shade_id,
                    "shade_name": row.shade_name,
                    "brand_name": row.brand_name,
                    "product_name": row.product_name,
                    "page_url": row.url,
                    "chip_url": row.chip_url,
                    "center_url_used": center_url_used,
                    "image_file": "",
                    "status": "error",
                    "error": error_msg,
                }
                writer.writerow(payload)
                err_writer.writerow(payload)
                print(f"[err] {row.product_id} {row.shade_id} -> {error_msg}", file=sys.stderr)

            if sleep_between > 0:
                time.sleep(float(sleep_between))

    print(f"[done] ok={ok_n} error={err_n}")
    print(f"[done] images_dir:  {out_dir.resolve().as_posix()}")
    print(f"[done] index_csv:   {index_csv_path.resolve().as_posix()}")
    print(f"[done] errors_csv:  {errors_csv_path.resolve().as_posix()}")
    return index_csv_path


def main() -> int:
    args = _parse_args()
    run(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        image_size=int(args.image_size),
        prefer=str(args.prefer),
        overwrite=bool(args.overwrite),
        timeout=int(args.timeout),
        retries=int(args.retries),
        sleep_between=float(args.sleep),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
