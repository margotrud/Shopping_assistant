# Scripts/build_color_lexicon.py
from __future__ import annotations

"""
Build an extended color lexicon (offline) and save it to JSON.

Goal:
- Produce: data/nlp/color_lexicon.json
- No dependency on inventory CSV.
- Cache all remote responses to disk.

IMPORTANT:
- The list *contents* are fetched from the distribution artifact of "color-name-lists"
  (not from api.color.pizza list-detail endpoints, which are not stable / often 404).
- We apply a small hygiene filter to exclude obvious product/non-color terms (e.g. "lipstick")
  to prevent false positives in mention extraction.

Usage:
    # Print available lists
    python Scripts/build_color_lexicon.py --print-lists

    # Build from lists
    python Scripts/build_color_lexicon.py --lists wikipedia xkcd ntc ral --include-webcolors --force-fetch --timeout 90
"""

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: requests. Install it in your venv.") from e


# ------------------------------------------------------------------------------
# Bootstrap: make src importable (project layout: pythonProject/ with src/)
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # pythonProject/
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
# Source-of-truth bundle for list contents (names + hex)
COLOR_NAME_LISTS_BLOB_URLS = [
    "https://unpkg.com/color-name-lists/dist/colorlists.json",
    "https://cdn.jsdelivr.net/npm/color-name-lists/dist/colorlists.json",
    "https://raw.githubusercontent.com/meodai/color-names/master/dist/colorlists.json",
]

DEFAULT_OUT = ROOT / "data" / "nlp" / "color_lexicon.json"
DEFAULT_CACHE_DIR = ROOT / "data" / "cache" / "color_lexicon"

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
SPACE_RE = re.compile(r"\s+")

# Hygiene: exclude obvious product / non-color terms that would create false positives.
# Keep this tight and stable (not a "color mapping").
DENY_KEYS: set[str] = {
    "lipstick",
    "lip stick",
    "lip",
    "lips",
    "gloss",
    "lipgloss",
    "makeup",
    "cosmetic",
    "cosmetics",
    "foundation",
    "concealer",
    "blush",
    "mascara",
    "eyeliner",
    "palette",
    "paint",
    "ink",
    "dye",
    "color",
    "colour",
}


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _norm_key(name: str) -> str:
    """
    Normalize a color-name key used for matching:
    - lowercase
    - remove punctuation (keep spaces)
    - collapse whitespace
    """
    s = name.strip().lower()
    s = re.sub(r"[_/\\\-]+", " ", s)
    s = re.sub(r"[^a-z0-9\s]+", "", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def _norm_hex(x: str) -> Optional[str]:
    if not x:
        return None
    x = x.strip()
    if not x.startswith("#"):
        x = "#" + x
    if HEX_RE.match(x):
        return x.lower()
    return None


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LexiconEntry:
    key: str
    name: str
    hex: str
    source: str


# ------------------------------------------------------------------------------
# HTTP client with disk cache
# ------------------------------------------------------------------------------
class CachedHTTP:
    def __init__(
        self,
        cache_dir: Path,
        *,
        sleep_s: float = 0.0,
        timeout_s: float = 25.0,
        user_agent: str = "ShoppingAssistantV8/ColorLexiconBuilder",
    ) -> None:
        self.cache_dir = cache_dir
        self.sleep_s = float(sleep_s)
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        _ensure_dir(cache_dir)

    def get_json(self, url: str, *, force: bool = False, retries: int = 2) -> Any:
        cache_path = self.cache_dir / f"{_sha1(url)}.json"
        if cache_path.exists() and not force:
            return _read_json(cache_path)

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                if self.sleep_s > 0:
                    time.sleep(self.sleep_s)

                r = self.session.get(url, timeout=self.timeout_s)
                r.raise_for_status()
                data = r.json()
                _write_json(cache_path, data)
                return data
            except Exception as e:
                last_err = e
                time.sleep(min(2.0 * (attempt + 1), 6.0))

        raise last_err  # type: ignore


# ------------------------------------------------------------------------------
# Color-name-lists blob (lists + entries)
# ------------------------------------------------------------------------------
def _fetch_colorlists_blob(http: CachedHTTP, *, force: bool = False, debug: bool = False) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for url in COLOR_NAME_LISTS_BLOB_URLS:
        try:
            data = http.get_json(url, force=force, retries=2)
            if not isinstance(data, dict):
                raise RuntimeError(f"blob unexpected type: {type(data)}")
            return data
        except Exception as e:
            last_err = e
            if debug:
                print("[build_color_lexicon] DEBUG blob fetch failed:", url, "->", repr(e))

    raise RuntimeError(f"Could not fetch colorlists.json from any mirror. Last error: {repr(last_err)}")


def fetch_available_lists(http: CachedHTTP, *, force: bool = False, debug: bool = False) -> List[str]:
    """
    Return list keys from the blob.
    Expected shapes:
      - {"lists": {"wikipedia": [...], "xkcd": [...], ...}, "meta": {...}}
      - or a direct mapping {"wikipedia": [...], ...}
    """
    blob = _fetch_colorlists_blob(http, force=force, debug=debug)

    if isinstance(blob.get("lists"), dict):
        keys = sorted(k for k in blob["lists"].keys() if isinstance(k, str))
        if keys:
            return keys

    bad = {"meta", "version", "readme", "info"}
    keys2 = sorted(
        k for k, v in blob.items()
        if isinstance(k, str) and k not in bad and isinstance(v, list)
    )
    if keys2:
        return keys2

    if debug:
        print("[build_color_lexicon] DEBUG blob keys:", sorted(blob.keys())[:80])
        if isinstance(blob.get("lists"), dict):
            print("[build_color_lexicon] DEBUG blob.lists keys:", sorted(blob["lists"].keys())[:80])

    raise RuntimeError("Could not extract list keys from color-name-lists blob.")


def fetch_list_colors(
    http: CachedHTTP,
    list_key: str,
    *,
    force: bool = False,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    blob = _fetch_colorlists_blob(http, force=force, debug=debug)

    if isinstance(blob.get("lists"), dict) and list_key in blob["lists"]:
        v = blob["lists"][list_key]
        if isinstance(v, list):
            return [it for it in v if isinstance(it, dict)]

    if list_key in blob and isinstance(blob[list_key], list):
        return [it for it in blob[list_key] if isinstance(it, dict)]

    if debug:
        print("[build_color_lexicon] DEBUG: list not found:", list_key)
        print("[build_color_lexicon] DEBUG blob keys:", sorted(blob.keys())[:80])
        if isinstance(blob.get("lists"), dict):
            print("[build_color_lexicon] DEBUG available lists:", sorted(blob["lists"].keys())[:80])

    raise RuntimeError(f"List '{list_key}' not found in color-name-lists blob.")


# ------------------------------------------------------------------------------
# Webcolors merge (optional)
# ------------------------------------------------------------------------------
def load_webcolors_css3_entries() -> List[LexiconEntry]:
    """
    Load CSS3 named colors from the 'webcolors' package, if available.
    """
    try:
        import webcolors  # type: ignore
    except Exception:
        return []

    mapping: Optional[Dict[str, str]] = None
    for attr in ("CSS3_NAMES_TO_HEX",):
        if hasattr(webcolors, attr):
            m = getattr(webcolors, attr)
            if isinstance(m, dict) and m:
                mapping = m
                break

    if mapping is None:
        try:
            names = list(webcolors.names("css3"))  # type: ignore
        except Exception:
            return []

        entries: List[LexiconEntry] = []
        for n in names:
            try:
                hx = webcolors.name_to_hex(n, spec="css3")  # type: ignore
            except Exception:
                continue
            key = _norm_key(n)
            if key in DENY_KEYS:
                continue
            nh = _norm_hex(hx)
            if not key or not nh:
                continue
            entries.append(LexiconEntry(key=key, name=n, hex=nh, source="webcolors:css3"))
        return entries

    entries2: List[LexiconEntry] = []
    for name, hx in mapping.items():
        key = _norm_key(name)
        if key in DENY_KEYS:
            continue
        nh = _norm_hex(hx)
        if not key or not nh:
            continue
        entries2.append(LexiconEntry(key=key, name=name, hex=nh, source="webcolors:css3"))
    return entries2


# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------
def build_lexicon(
    http: CachedHTTP,
    list_keys: List[str],
    *,
    include_webcolors: bool = False,
    force_fetch: bool = False,
    debug: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
        dict: normalized_key -> {name, hex, source}
    """
    entries: List[LexiconEntry] = []

    for lk in list_keys:
        colors = fetch_list_colors(http, lk, force=force_fetch, debug=debug)
        for c in colors:
            name = c.get("name") or c.get("title") or c.get("label")
            hx = c.get("hex") or c.get("value") or c.get("colour")
            if not isinstance(name, str) or not isinstance(hx, str):
                continue

            key = _norm_key(name)
            if not key or key in DENY_KEYS:
                continue

            nh = _norm_hex(hx)
            if not nh:
                continue

            entries.append(
                LexiconEntry(
                    key=key,
                    name=name.strip(),
                    hex=nh,
                    source=f"color-name-lists:{lk}",
                )
            )

    if include_webcolors:
        entries.extend(load_webcolors_css3_entries())

    def source_rank(src: str) -> int:
        if src.startswith("color-name-lists:"):
            return 0
        if src.startswith("webcolors:"):
            return 1
        return 9

    entries_sorted = sorted(entries, key=lambda e: (source_rank(e.source), e.key, e.name))
    out: Dict[str, Dict[str, Any]] = {}
    for e in entries_sorted:
        if e.key in out:
            continue
        out[e.key] = {"name": e.name, "hex": e.hex, "source": e.source}

    return out


def write_lexicon_json(path: Path, lex: Dict[str, Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    ordered = {k: lex[k] for k in sorted(lex.keys())}
    with path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build extended color lexicon JSON (color-name-lists + optional webcolors).")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT), help=f"Output JSON path (default: {DEFAULT_OUT})")
    p.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help=f"Cache directory (default: {DEFAULT_CACHE_DIR})")
    p.add_argument("--lists", nargs="*", default=[], help="List keys to fetch (use --print-lists to see keys).")
    p.add_argument("--print-lists", action="store_true", help="Print available list keys and exit.")
    p.add_argument("--include-webcolors", action="store_true", help="Merge CSS3 named colors from webcolors (if installed).")
    p.add_argument("--force-fetch", action="store_true", help="Ignore cache and re-fetch remote payloads.")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between HTTP requests (rate-limit friendly).")
    p.add_argument("--debug-payload", action="store_true", help="Print raw payload keys/samples on schema errors.")
    p.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds (default: 60).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    cache_dir = Path(args.cache_dir)

    http = CachedHTTP(
        cache_dir=cache_dir,
        sleep_s=float(args.sleep),
        timeout_s=float(args.timeout),
    )

    if args.print_lists:
        keys = fetch_available_lists(http, force=args.force_fetch, debug=args.debug_payload)
        for k in keys:
            print(k)
        return 0

    if not args.lists:
        available = fetch_available_lists(http, force=args.force_fetch, debug=args.debug_payload)
        if not available:
            raise RuntimeError("No lists available from color-name-lists blob.")
        default = ["wikipedia", "xkcd", "ntc", "ral"]
        list_keys = default if all(k in available for k in default) else available[:1]
        print(f"[build_color_lexicon] No --lists provided. Using: {list_keys}")
    else:
        list_keys = list(args.lists)

    try:
        available = fetch_available_lists(http, force=False, debug=args.debug_payload)
    except Exception:
        available = []

    if available:
        unknown = [k for k in list_keys if k not in available]
        if unknown:
            print(f"[build_color_lexicon] WARNING: unknown list keys: {unknown}")
            print(
                "[build_color_lexicon] Known keys include: "
                f"{available[:30]}{' ...' if len(available) > 30 else ''}"
            )

    lex = build_lexicon(
        http,
        list_keys=list_keys,
        include_webcolors=bool(args.include_webcolors),
        force_fetch=bool(args.force_fetch),
        debug=bool(args.debug_payload),
    )

    write_lexicon_json(out_path, lex)
    print(f"[build_color_lexicon] Wrote {len(lex)} entries -> {out_path}")
    print(f"[build_color_lexicon] Cache dir: {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
