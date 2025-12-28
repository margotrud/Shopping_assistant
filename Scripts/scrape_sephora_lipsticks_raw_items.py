from __future__ import annotations

import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


# =============================================================================
# Paths / Output
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = PROJECT_ROOT / "data" / "Sephora_lipsticks_raw_items.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "product_id",
    "shade_id",
    "brand_name",
    "product_name",
    "shade_name",
    "url",
    "chip_url",
]

# =============================================================================
# Target category
# =============================================================================

SEPHORA_BASE = "https://www.sephora.fr"
CATEGORY_URL = "https://www.sephora.fr/shop/maquillage/levres/rouge-a-levres-c371/"


# =============================================================================
# HTTP
# =============================================================================

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": UA,
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": SEPHORA_BASE + "/",
}

TIMEOUT_S = 35


def _dbg(*a: object) -> None:
    print("[dbg]", *a, file=sys.stderr)


def _sleep(i: int, base: float) -> None:
    time.sleep(base + (i % 3) * 0.15)


def _norm(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = " ".join(str(x).split()).strip()
    return s or None


def _abs_url(href: str) -> str:
    return urljoin(SEPHORA_BASE, href)


def _get_html(sess: requests.Session, url: str) -> str:
    r = sess.get(url, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.text


# =============================================================================
# Regex
# =============================================================================

PRODUCT_ID_RE = re.compile(r"-(P\d+)\.html", re.IGNORECASE)

# chip urls typically look like:
# https://media.sephora.eu/content/dam/digital/pim/published/S/SEPHORA_COLLECTION/123/456789-media_thumbnail.jpg
CHIP_URL_RE = re.compile(r"^https://media\.sephora\.eu/.*-media_thumbnail\.jpg", re.IGNORECASE)
SHADE_ID_FROM_CHIP_RE = re.compile(r"/(\d+)-media_thumbnail\.jpg", re.IGNORECASE)

# brand in chip url
BRAND_FROM_CHIP_RE = re.compile(r"/published/[A-Z]/([^/]+)/", re.IGNORECASE)


def _product_id_from_url(url: str) -> Optional[str]:
    m = PRODUCT_ID_RE.search(url)
    return m.group(1).upper() if m else None


def _shade_id_from_chip_url(chip_url: str) -> Optional[str]:
    m = SHADE_ID_FROM_CHIP_RE.search(chip_url.split("?", 1)[0])
    return m.group(1) if m else None


def _brand_from_chip_url(chip_url: str) -> Optional[str]:
    m = BRAND_FROM_CHIP_RE.search(chip_url)
    return _norm(m.group(1)) if m else None


# =============================================================================
# JSON walkers
# =============================================================================

def _iter_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_dicts(v)


def _first_str(*vals: Any) -> Optional[str]:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


# =============================================================================
# Category: collect PDP urls
# =============================================================================

def _extract_pdp_urls_from_category_html(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: List[str] = []

    for a in soup.select('a[href*="/p/"]'):
        href = (a.get("href") or "").strip()
        if not href or ".html" not in href:
            continue
        u = _abs_url(href)
        if _product_id_from_url(u):
            urls.append(u)

    # dedup stable
    seen = set()
    out: List[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def collect_all_pdp_urls(sess: requests.Session, cfg: "ScrapeConfig") -> List[str]:
    all_urls: List[str] = []
    seen = set()

    for page in range(1, cfg.max_pages + 1):
        page_url = CATEGORY_URL if page == 1 else f"{CATEGORY_URL}?page={page}"
        try:
            html = _get_html(sess, page_url)
        except Exception as e:
            _dbg("category fetch failed", page_url, repr(e))
            break

        pdps = _extract_pdp_urls_from_category_html(html)
        if not pdps:
            _dbg("no pdps found -> stop at page", page)
            break

        added = 0
        for u in pdps:
            if u in seen:
                continue
            seen.add(u)
            all_urls.append(u)
            added += 1

        _dbg(f"category page={page} +{added} total={len(all_urls)}")
        _sleep(page, cfg.sleep_s_base)

        if added == 0 and page >= 2:
            break

    return all_urls


# =============================================================================
# PDP: extract product_name / brand_name (source HTML)
# =============================================================================

def _parse_next_data(soup: BeautifulSoup) -> Optional[dict]:
    tag = soup.select_one("script#__NEXT_DATA__")
    if not tag:
        return None
    raw = tag.get_text(strip=True)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _parse_json_ld_product(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (brand_name, product_name) from JSON-LD Product, if present.
    """
    for tag in soup.select('script[type="application/ld+json"]'):
        raw = tag.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        nodes = data if isinstance(data, list) else [data]
        for node in nodes:
            if not isinstance(node, dict):
                continue
            t = node.get("@type")
            is_product = (t == "Product") or (isinstance(t, list) and "Product" in t)
            if not is_product:
                continue

            product_name = _norm(node.get("name"))
            brand_name = None
            b = node.get("brand")
            if isinstance(b, dict):
                brand_name = _norm(b.get("name"))
            elif isinstance(b, str):
                brand_name = _norm(b)

            if brand_name or product_name:
                return (brand_name, product_name)

    return (None, None)


def _dom_brand_product(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str]]:
    """
    Fallback: product_name from h1, brand_name from closest short text above it.
    """
    h1 = soup.select_one("h1")
    product_name = _norm(h1.get_text(" ", strip=True)) if h1 else None

    brand = None
    if h1:
        for el in h1.find_all_previous(["a", "div", "span"], limit=80):
            t = _norm(el.get_text(" ", strip=True))
            if not t or t == product_name:
                continue
            # brand blocks are short
            if 2 <= len(t) <= 60:
                brand = t
                break

    return (brand, product_name)


def _extract_brand_product(soup: BeautifulSoup, next_data: Optional[dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Priority:
      1) __NEXT_DATA__ (if has product node)
      2) JSON-LD Product
      3) DOM fallback
    """
    brand_nd = None
    name_nd = None

    if next_data:
        # try canonical path, else search
        product = None
        try:
            product = next_data["props"]["pageProps"]["product"]
        except Exception:
            pass

        if isinstance(product, dict):
            name_nd = _norm(product.get("displayName") or product.get("productName") or product.get("name") or product.get("title"))
            b = product.get("brand")
            if isinstance(b, dict):
                brand_nd = _norm(b.get("name"))
            elif isinstance(b, str):
                brand_nd = _norm(b)
            if not brand_nd:
                brand_nd = _norm(product.get("brandName"))
        else:
            # fallback: search any dict
            for d in _iter_dicts(next_data):
                n = _norm(d.get("displayName") or d.get("productName") or d.get("name") or d.get("title"))
                if not n:
                    continue
                b = d.get("brand")
                bname = None
                if isinstance(b, dict):
                    bname = _norm(b.get("name"))
                elif isinstance(b, str):
                    bname = _norm(b)
                if not bname:
                    bname = _norm(d.get("brandName"))
                if n and bname:
                    name_nd = n
                    brand_nd = bname
                    break

    brand_ld, name_ld = _parse_json_ld_product(soup)
    brand_dom, name_dom = _dom_brand_product(soup)

    brand = _first_str(brand_nd, brand_ld, brand_dom)
    name = _first_str(name_nd, name_ld, name_dom)
    return (_norm(brand), _norm(name))


# =============================================================================
# PDP: extract shades (source HTML)
# =============================================================================

def _extract_shades_from_next_data(next_data: dict) -> List[Dict[str, str]]:
    """
    Best case: product['skus'] contains skuId/variationValue/swatchImage.
    Fallback: walk the json looking for dicts with swatchImage-like fields.
    """
    out: List[Dict[str, str]] = []
    seen = set()

    product = None
    try:
        product = next_data["props"]["pageProps"]["product"]
    except Exception:
        product = None

    if isinstance(product, dict) and isinstance(product.get("skus"), list):
        for sku in product["skus"]:
            if not isinstance(sku, dict):
                continue
            shade_id = _norm(sku.get("skuId") or sku.get("shadeId") or sku.get("id"))
            shade_name = _norm(sku.get("variationValue") or sku.get("shadeName") or sku.get("name") or sku.get("label"))
            chip_url = _norm(sku.get("swatchImage") or sku.get("swatchUrl") or sku.get("swatch"))
            if not (shade_id and shade_name and chip_url):
                continue
            if not CHIP_URL_RE.match(chip_url):
                continue
            k = (shade_id, chip_url)
            if k in seen:
                continue
            seen.add(k)
            out.append({"shade_id": shade_id, "shade_name": shade_name, "chip_url": chip_url})
        if out:
            return out

    # generic walk fallback
    for d in _iter_dicts(next_data):
        chip_url = _norm(d.get("swatchImage") or d.get("swatchUrl") or d.get("swatch"))
        if not (chip_url and CHIP_URL_RE.match(chip_url)):
            continue
        shade_id = _norm(d.get("skuId") or d.get("shadeId") or d.get("id") or d.get("variantId"))
        shade_name = _norm(d.get("variationValue") or d.get("shadeName") or d.get("name") or d.get("label") or d.get("value"))

        if not (shade_name and chip_url):
            continue

        # repair shade_id from chip filename if missing or non-digit
        sid2 = _shade_id_from_chip_url(chip_url)
        if sid2:
            shade_id = sid2
        if not shade_id:
            continue

        k = (shade_id, chip_url)
        if k in seen:
            continue
        seen.add(k)
        out.append({"shade_id": shade_id, "shade_name": shade_name, "chip_url": chip_url})

    return out


def _extract_shades_from_html(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Fallback if __NEXT_DATA__ missing:
    find chip <img> urls in HTML: media_thumbnail.jpg
    shade_id from filename; shade_name from alt/title if present.
    """
    out: List[Dict[str, str]] = []
    seen = set()

    for img in soup.find_all("img"):
        src = (img.get("src") or "").strip()
        if not src or not CHIP_URL_RE.match(src):
            continue
        shade_id = _shade_id_from_chip_url(src)
        if not shade_id:
            continue
        shade_name = _norm(img.get("alt") or img.get("title"))
        if not shade_name:
            # keep but mark; will be filtered later if required
            shade_name = None

        k = (shade_id, src)
        if k in seen:
            continue
        seen.add(k)
        out.append({"shade_id": shade_id, "shade_name": shade_name or "", "chip_url": src})

    return out


# =============================================================================
# PDP: build rows
# =============================================================================

def scrape_pdp(sess: requests.Session, url: str) -> List[Dict[str, str]]:
    product_id = _product_id_from_url(url)
    if not product_id:
        raise RuntimeError(f"cannot parse product_id from url: {url}")

    html = _get_html(sess, url)
    soup = BeautifulSoup(html, "html.parser")
    next_data = _parse_next_data(soup)

    brand_name, product_name = _extract_brand_product(soup, next_data)

    # shades
    shades: List[Dict[str, str]] = []
    if next_data:
        shades = _extract_shades_from_next_data(next_data)
    if not shades:
        shades = _extract_shades_from_html(soup)

    # if still no brand -> last resort from any chip url
    if not brand_name:
        for sh in shades:
            b = _brand_from_chip_url(sh.get("chip_url", ""))
            if b:
                brand_name = b
                break

    # hard gate: required product fields
    if not brand_name or not product_name:
        # without these, row is unusable
        return []

    rows: List[Dict[str, str]] = []
    seen = set()

    for sh in shades:
        shade_id = _norm(sh.get("shade_id"))
        chip_url = _norm(sh.get("chip_url"))
        shade_name = _norm(sh.get("shade_name")) or ""

        if not (shade_id and chip_url and CHIP_URL_RE.match(chip_url)):
            continue

        # repair shade_id from chip filename (canonical)
        sid2 = _shade_id_from_chip_url(chip_url)
        if sid2:
            shade_id = sid2

        # some fallbacks might have empty shade_name; try to keep only if present
        if not shade_name:
            continue

        key = (product_id, shade_id)
        if key in seen:
            continue
        seen.add(key)

        rows.append(
            {
                "product_id": product_id,
                "shade_id": shade_id,
                "brand_name": brand_name,
                "product_name": product_name,
                "shade_name": shade_name,
                "url": url,
                "chip_url": chip_url,
            }
        )

    return rows


# =============================================================================
# Output helpers
# =============================================================================

def dedupe_combo(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for r in rows:
        key = (r.get("product_id"), r.get("shade_id"))
        if not key[0] or not key[1]:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


# =============================================================================
# Main
# =============================================================================

@dataclass(frozen=True)
class ScrapeConfig:
    max_pages: int = 250
    sleep_s_base: float = 0.55
    max_pdps: Optional[int] = None  # for debugging


def main() -> None:
    cfg = ScrapeConfig()

    sess = requests.Session()
    sess.headers.update(HEADERS)

    # Warm-up: hit base + category once to get basic cookies
    try:
        _get_html(sess, SEPHORA_BASE + "/")
        _get_html(sess, CATEGORY_URL)
    except Exception:
        pass

    pdp_urls = collect_all_pdp_urls(sess, cfg)
    if cfg.max_pdps is not None:
        pdp_urls = pdp_urls[: cfg.max_pdps]

    _dbg("pdp_urls =", len(pdp_urls))

    all_rows: List[Dict[str, str]] = []
    fail = 0
    zero = 0

    for i, url in enumerate(pdp_urls, start=1):
        try:
            rows = scrape_pdp(sess, url)
            if not rows:
                zero += 1
            all_rows.extend(rows)
        except Exception as e:
            fail += 1
            _dbg("pdp fail", f"{i}/{len(pdp_urls)}", url, repr(e))

        if i % 25 == 0:
            _dbg("progress", f"{i}/{len(pdp_urls)}", "rows=", len(all_rows), "fail=", fail, "zero_products=", zero)

        _sleep(i, cfg.sleep_s_base)

    all_rows = dedupe_combo(all_rows)
    write_csv(all_rows, OUT_CSV)

    _dbg("done rows=", len(all_rows), "fail=", fail, "zero_products=", zero)
    print(f"[ok] wrote {len(all_rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
