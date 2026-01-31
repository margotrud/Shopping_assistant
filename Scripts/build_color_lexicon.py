# Scripts/build_color_lexicon.py
from __future__ import annotations

"""
Build an extended colors lexicon (offline) and save it to JSON.

Goal:
- Produce: data/nlp/color_lexicon.json
- No dependency on inventory CSV.
- Cache all remote responses to disk.

IMPORTANT:
- The list *contents* are fetched from the distribution artifact of "colors-name-lists"
  (not from api.colors.pizza list-detail endpoints, which are not stable / often 404).
- Apply a small hygiene filter to exclude obvious product/non-colors terms to prevent false positives.
- Descriptors (e.g. "nude", "pastel", "neon") are derived for REPORT only (not written into color_lexicon.json).

NEW (data-driven fix):
- Drop "modifier-like" single-token keys (e.g. "dark") when they mostly occur as prefix
  of multi-token colors names ("dark rose", "dark mauve", ...), to prevent resolve("dark")
  returning a compound-derived hex.
  This is done via structural stats (no hardcoded word list).
"""
import os
import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
COLOR_NAME_LISTS_BLOB_URLS = [
    "https://unpkg.com/color-name-lists/dist/colorlists.json",
    "https://cdn.jsdelivr.net/npm/color-name-lists/dist/colorlists.json",
    "https://raw.githubusercontent.com/meodai/color-names/master/dist/colorlists.json",
]

DEFAULT_OUT = ROOT / "data" / "nlp" / "color_lexicon.json"
DEFAULT_REPORT = ROOT / "data" / "nlp" / "color_lexicon_report.json"
DEFAULT_CACHE_DIR = ROOT / "data" / "cache" / "color_lexicon"

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9]+")

# Hygiene: exclude obvious product / non-colors terms.
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
    "colors",
    "colour",
}

# ------------------------------------------------------------------------------
# Generic stabilization policy (no inventory dependency)
# ------------------------------------------------------------------------------
BASE_ANCHOR_TRIGGER_MIN_C = 35.0
BASE_ANCHOR_TRIGGER_MAX_L = 78.0
BASE_ANCHOR_VARIANT_MIN_N = 8
BASE_ANCHOR_VARIANT_MAX_L = 80.0

# If the family is "neutral-ish" (median chroma low), do NOT pick top-chroma tranche
# because it drifts toward olive/brown extremes; instead pick around median chroma.
NEUTRAL_FAMILY_MEDIAN_C_MAX = 28.0
NEUTRAL_KEEP_C_QLO = 0.30
NEUTRAL_KEEP_C_QHI = 0.70

# Neutral anchor selection: target medians rather than global medoid to avoid drift
NEUTRAL_TARGET_SELECT_TOPK = 25
NEUTRAL_TARGET_W_C = 0.80

# Descriptor derivation (e.g. nude, pastel, neon...) -> report only
DESCRIPTOR_MIN_PHRASE_KEYS = 10
DESCRIPTOR_MIN_UNIQUE_HEX = 6
DESCRIPTOR_MAX_TOKEN_LEN = 16

# Neutral-ish descriptor anchor filter
DESCRIPTOR_MAX_L = 85.0  # avoid near-white
DESCRIPTOR_MIN_L = 35.0  # avoid very dark
DESCRIPTOR_MAX_C = 45.0  # keep reasonably neutral (prevents "nude"->hot pink)
DESCRIPTOR_KEEP_C_QHI = 0.75  # robust keep (drop top 25% chroma)

# ------------------------------------------------------------------------------
# NEW: modifier-singleton suppression (data-driven, no hardcoded tokens)
# ------------------------------------------------------------------------------
# Thresholds are intentionally conservative and can be tuned via env if needed.
MOD_SINGLE_MIN_PHRASE_KEYS = int(
    re.sub(r"[^0-9]", "", str(os.environ.get("SA_LEX_DROP_MOD_SINGLE_MIN_PHRASE_KEYS", "18"))) or "18"
)
MOD_SINGLE_MIN_PREFIX_KEYS = int(
    re.sub(r"[^0-9]", "", str(os.environ.get("SA_LEX_DROP_MOD_SINGLE_MIN_PREFIX_KEYS", "14"))) or "14"
)
MOD_SINGLE_MIN_PREFIX_FRAC = float(os.environ.get("SA_LEX_DROP_MOD_SINGLE_MIN_PREFIX_FRAC", "0.80"))
MOD_SINGLE_MAX_SINGLE_CONF = int(
    re.sub(r"[^0-9]", "", str(os.environ.get("SA_LEX_DROP_MOD_SINGLE_MAX_SINGLE_CONF", "2"))) or "2"
)
MOD_SINGLE_MAX_N_HEX = int(re.sub(r"[^0-9]", "", str(os.environ.get("SA_LEX_DROP_MOD_SINGLE_MAX_N_HEX", "2"))) or "2")
MOD_SINGLE_MAX_N_SOURCES = int(
    re.sub(r"[^0-9]", "", str(os.environ.get("SA_LEX_DROP_MOD_SINGLE_MAX_N_SOURCES", "2"))) or "2"
)


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
    s = name.strip().lower()
    s = re.sub(r"[_/\\\-]+", " ", s)
    s = re.sub(r"[^a-z0-9\s]+", "", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def _tokens(key: str) -> List[str]:
    return TOKEN_RE.findall(key.lower())


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


def _hex_to_rgb01(hx: str) -> Optional[Tuple[float, float, float]]:
    hx = str(hx or "").strip().lower()
    if not hx:
        return None
    if not hx.startswith("#"):
        hx = "#" + hx
    if not HEX_RE.match(hx):
        return None
    r = int(hx[1:3], 16) / 255.0
    g = int(hx[3:5], 16) / 255.0
    b = int(hx[5:7], 16) / 255.0
    return (r, g, b)


def _srgb01_to_linear(x: float) -> float:
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def _hex_to_lab(hx: str) -> Optional[Tuple[float, float, float]]:
    rgb = _hex_to_rgb01(hx)
    if rgb is None:
        return None
    r, g, b = rgb
    r, g, b = _srgb01_to_linear(r), _srgb01_to_linear(g), _srgb01_to_linear(b)

    # sRGB D65 -> XYZ
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # D65 white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = X / Xn, Y / Yn, Z / Zn

    d = 6 / 29

    def f(t: float) -> float:
        return t ** (1 / 3) if t > d**3 else (t / (3 * d**2) + 4 / 29)

    fx, fy, fz = f(float(x)), f(float(y)), f(float(z))
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b2 = 200 * (fy - fz)
    return (float(L), float(a), float(b2))


def _lab_chroma(lab: Tuple[float, float, float]) -> float:
    _, a, b = lab
    return float((a * a + b * b) ** 0.5)


def _dist2(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _quantile(vals: List[float], q: float) -> float:
    v = [float(x) for x in vals if x is not None]
    v.sort()
    if not v:
        return float("nan")
    if q <= 0:
        return v[0]
    if q >= 1:
        return v[-1]
    idx = q * (len(v) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(v) - 1)
    w = idx - lo
    return (1 - w) * v[lo] + w * v[hi]


def _lab_medoid_hex(hexes: List[str]) -> Optional[Tuple[str, Dict[str, float]]]:
    labs: List[Tuple[str, Tuple[float, float, float], float, float]] = []
    for hx in hexes:
        lab = _hex_to_lab(hx)
        if lab is None:
            continue
        L = float(lab[0])
        C = float(_lab_chroma(lab))
        labs.append((hx, lab, L, C))
    if not labs:
        return None

    best_hx = None
    best_sum = float("inf")
    for i in range(len(labs)):
        _, li, _, _ = labs[i]
        s = 0.0
        for j in range(len(labs)):
            if i == j:
                continue
            _, lj, _, _ = labs[j]
            s += _dist2(li, lj)
        if s < best_sum:
            best_sum = s
            best_hx = labs[i][0]

    if best_hx is None:
        return None

    lab0 = _hex_to_lab(best_hx)
    if lab0 is None:
        return None
    meta = {"L": float(lab0[0]), "C": float(_lab_chroma(lab0))}
    return best_hx, meta


def _count_phrase_keys_for_bases(key_index: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """
    For each base token t, count how many multi-token keys contain t.
    This is a structural signal that t is a "family" token (e.g. berry),
    even if its single-token entry has low n_sources/n_hex.
    """
    counts: Dict[str, int] = {}
    for k in key_index.keys():
        toks = _tokens(k)
        if len(toks) < 2:
            continue
        for t in set(toks):
            if t and t.isalpha():
                counts[t] = counts.get(t, 0) + 1
    return counts


def _count_prefix_keys_for_tokens(key_index: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """
    For each token t, count how many multi-token keys start with t.
    Strong signal of "modifier-like" usage (e.g. dark rose, dark mauve, dark berry...).
    """
    counts: Dict[str, int] = {}
    for k in key_index.keys():
        toks = _tokens(k)
        if len(toks) < 2:
            continue
        t0 = toks[0]
        if t0 and t0.isalpha():
            counts[t0] = counts.get(t0, 0) + 1
    return counts


def _drop_modifier_like_singletons(
    out: Dict[str, Dict[str, Any]],
    phrase_counts: Dict[str, int],
    prefix_counts: Dict[str, int],
) -> Dict[str, Any]:
    """
    Remove single-token keys that behave like modifiers (structurally) rather than colors.

    Data-driven policy:
      - token appears in many multi-token names, especially as first token
      - and has low single confidence / low source+hex support
    """
    dropped: Dict[str, Any] = {}

    for k in list(out.keys()):
        if " " in k:
            continue
        if k in DENY_KEYS:
            continue

        nphr = int(phrase_counts.get(k, 0))
        npfx = int(prefix_counts.get(k, 0))
        if nphr < MOD_SINGLE_MIN_PHRASE_KEYS:
            continue
        if npfx < MOD_SINGLE_MIN_PREFIX_KEYS:
            continue

        frac = (float(npfx) / float(max(1, nphr))) if nphr > 0 else 0.0
        if frac < MOD_SINGLE_MIN_PREFIX_FRAC:
            continue

        info = out.get(k, {})
        sc = int(info.get("single_confidence", 0) or 0)
        ns = int(info.get("n_sources", 0) or 0)
        nh = int(info.get("n_hex", 0) or 0)

        # "true base colors" usually have decent support; keep them.
        # modifier-like garbage tends to be low-support singletons.
        if sc > MOD_SINGLE_MAX_SINGLE_CONF:
            continue
        if ns > MOD_SINGLE_MAX_N_SOURCES:
            continue
        if nh > MOD_SINGLE_MAX_N_HEX:
            continue

        dropped[k] = {
            "reason": "modifier-like-singleton",
            "n_phrase_keys": nphr,
            "n_prefix_keys": npfx,
            "prefix_frac": frac,
            "n_sources": ns,
            "n_hex": nh,
            "single_confidence": sc,
            "hex": info.get("hex"),
        }
        out.pop(k, None)

    return dropped


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
    blob = _fetch_colorlists_blob(http, force=force, debug=debug)

    if isinstance(blob.get("lists"), dict):
        keys = sorted(k for k in blob["lists"].keys() if isinstance(k, str))
        if keys:
            return keys

    bad = {"meta", "version", "readme", "info"}
    keys2 = sorted(k for k, v in blob.items() if isinstance(k, str) and k not in bad and isinstance(v, list))
    if keys2:
        return keys2

    if debug:
        print("[build_color_lexicon] DEBUG blob keys:", sorted(blob.keys())[:80])
        if isinstance(blob.get("lists"), dict):
            print("[build_color_lexicon] DEBUG blob.lists keys:", sorted(blob["lists"].keys())[:80])

    raise RuntimeError("Could not extract list keys from colors-name-lists blob.")


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

    raise RuntimeError(f"List '{list_key}' not found in colors-name-lists blob.")


# ------------------------------------------------------------------------------
# Webcolors merge (optional)
# ------------------------------------------------------------------------------
def load_webcolors_css3_entries() -> List[LexiconEntry]:
    try:
        import webcolors  # type: ignore
    except Exception:
        return []

    mapping: Optional[Dict[str, str]] = None
    if hasattr(webcolors, "CSS3_NAMES_TO_HEX"):
        m = getattr(webcolors, "CSS3_NAMES_TO_HEX")
        if isinstance(m, dict) and m:
            mapping = m

    entries: List[LexiconEntry] = []
    if mapping is None:
        try:
            names = list(webcolors.names("css3"))  # type: ignore
        except Exception:
            return []
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

    for name, hx in mapping.items():
        key = _norm_key(name)
        if key in DENY_KEYS:
            continue
        nh = _norm_hex(hx)
        if not key or not nh:
            continue
        entries.append(LexiconEntry(key=key, name=name, hex=nh, source="webcolors:css3"))
    return entries


# ------------------------------------------------------------------------------
# Anchor selection per key (use all hex for that key)
# ------------------------------------------------------------------------------
def _build_key_index(entries: List[LexiconEntry]) -> Dict[str, Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        d = by_key.setdefault(
            e.key,
            {
                "names": [],
                "hexes": [],
                "sources": [],
            },
        )
        d["names"].append(e.name)
        d["hexes"].append(e.hex)
        d["sources"].append(e.source)
    return by_key


def _choose_anchor_for_key(key: str, info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    hexes_raw = [h for h in info.get("hexes", []) if isinstance(h, str)]
    hexes: List[str] = []
    seen: set[str] = set()
    for h in hexes_raw:
        nh = _norm_hex(h)
        if not nh or nh in seen:
            continue
        seen.add(nh)
        hexes.append(nh)

    if not hexes:
        return None

    chosen = _lab_medoid_hex(hexes)
    if chosen is None:
        return None
    anchor_hex, meta = chosen

    names = [n for n in info.get("names", []) if isinstance(n, str)]
    name = names[0] if names else key

    sources = [s for s in info.get("sources", []) if isinstance(s, str)]
    n_sources = len(set(sources))

    return {
        "name": name,
        "hex": anchor_hex,
        "anchor_policy": "lab-medoid",
        "n_hex": len(hexes),
        "n_sources": n_sources,
        "L_anchor": float(meta["L"]),
        "C_anchor": float(meta["C"]),
    }


# ------------------------------------------------------------------------------
# Base-anchor stabilization (generic, neutral-aware)
# ------------------------------------------------------------------------------
def _stabilize_base_anchors(out: Dict[str, Dict[str, Any]], key_index: Dict[str, Dict[str, Any]]) -> None:
    keys = list(out.keys())
    for base in keys:
        if base in DENY_KEYS or " " in base:
            continue
        cur_hex = out.get(base, {}).get("hex")
        if not isinstance(cur_hex, str):
            continue
        cur_lab = _hex_to_lab(cur_hex)
        if cur_lab is None:
            continue

        L0 = float(cur_lab[0])
        C0 = float(_lab_chroma(cur_lab))
        if (C0 >= float(BASE_ANCHOR_TRIGGER_MIN_C)) and (L0 <= float(BASE_ANCHOR_TRIGGER_MAX_L)):
            continue

        pat = re.compile(rf"(?:^|\s){re.escape(base)}(?:\s|$)")
        cand_hex: List[str] = []
        seen_hex: set[str] = set()

        for k, info in key_index.items():
            if " " not in k:
                continue
            if not pat.search(k):
                continue
            for h in info.get("hexes", []):
                if not isinstance(h, str):
                    continue
                nh = _norm_hex(h)
                if not nh or nh in seen_hex:
                    continue
                seen_hex.add(nh)
                cand_hex.append(nh)

        if len(cand_hex) < int(BASE_ANCHOR_VARIANT_MIN_N):
            continue

        labs: List[Tuple[str, Tuple[float, float, float], float, float]] = []
        for hx in cand_hex:
            lab = _hex_to_lab(hx)
            if lab is None:
                continue
            L = float(lab[0])
            C = float(_lab_chroma(lab))
            if L > float(BASE_ANCHOR_VARIANT_MAX_L):
                continue
            labs.append((hx, lab, L, C))

        if len(labs) < int(BASE_ANCHOR_VARIANT_MIN_N):
            continue

        cvals = [x[3] for x in labs]
        lvals = [x[2] for x in labs]
        c_med = _quantile(cvals, 0.50)
        L_med = _quantile(lvals, 0.50)

        # neutral family: keep mid-chroma band
        if float(c_med) <= float(NEUTRAL_FAMILY_MEDIAN_C_MAX):
            c_lo = _quantile(cvals, float(NEUTRAL_KEEP_C_QLO))
            c_hi = _quantile(cvals, float(NEUTRAL_KEEP_C_QHI))
            labs2 = [x for x in labs if (x[3] >= c_lo and x[3] <= c_hi)]
            if len(labs2) >= int(BASE_ANCHOR_VARIANT_MIN_N):
                labs = labs2

            scored = sorted(
                labs,
                key=lambda x: abs(x[2] - float(L_med)) + float(NEUTRAL_TARGET_W_C) * abs(x[3] - float(c_med)),
            )
            topk = int(NEUTRAL_TARGET_SELECT_TOPK)
            shortlist = scored[: max(5, min(topk, len(scored)))]
            chosen = _lab_medoid_hex([x[0] for x in shortlist])
        else:
            c_thr = _quantile(cvals, 0.60)
            labs2 = [x for x in labs if x[3] >= c_thr]
            if len(labs2) >= int(BASE_ANCHOR_VARIANT_MIN_N):
                labs = labs2
            chosen = _lab_medoid_hex([x[0] for x in labs])

        if chosen is None:
            continue
        best_hx, meta = chosen

        out[base]["hex"] = best_hx
        out[base]["anchor_policy"] = "derived-from-variants"
        out[base]["L_anchor"] = float(meta["L"])
        out[base]["C_anchor"] = float(meta["C"])


# ------------------------------------------------------------------------------
# Descriptor derivation (generic): report-only (do not write into out)
# ------------------------------------------------------------------------------
def _derive_descriptors_report_only(
    key_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    token_counts: Dict[str, int] = {}
    token_in_phrases: Dict[str, List[str]] = {}

    for k in key_index.keys():
        toks = _tokens(k)
        if len(toks) < 2:
            continue
        for t in toks:
            if not t:
                continue
            if not t.isalpha():
                continue
            if len(t) > DESCRIPTOR_MAX_TOKEN_LEN:
                continue
            token_counts[t] = token_counts.get(t, 0) + 1
            token_in_phrases.setdefault(t, []).append(k)

    derived: Dict[str, Any] = {}

    for t, nphr in sorted(token_counts.items(), key=lambda x: (-x[1], x[0])):
        if t in DENY_KEYS:
            continue
        if nphr < int(DESCRIPTOR_MIN_PHRASE_KEYS):
            continue

        keys = token_in_phrases.get(t, [])
        seen_hex: set[str] = set()
        hexes: List[str] = []
        seen_src: set[str] = set()

        for k in keys:
            info = key_index.get(k, {})

            for s in info.get("sources", []):
                if not isinstance(s, str):
                    continue
                seen_src.add(s)

            for h in info.get("hexes", []):
                if not isinstance(h, str):
                    continue
                nh = _norm_hex(h)
                if not nh or nh in seen_hex:
                    continue
                seen_hex.add(nh)
                hexes.append(nh)

        if len(hexes) < int(DESCRIPTOR_MIN_UNIQUE_HEX):
            continue

        labs: List[Tuple[str, Tuple[float, float, float], float, float]] = []
        for hx in hexes:
            lab = _hex_to_lab(hx)
            if lab is None:
                continue
            L = float(lab[0])
            C = float(_lab_chroma(lab))
            if L < float(DESCRIPTOR_MIN_L) or L > float(DESCRIPTOR_MAX_L):
                continue
            if C > float(DESCRIPTOR_MAX_C):
                continue
            labs.append((hx, lab, L, C))

        if len(labs) < int(DESCRIPTOR_MIN_UNIQUE_HEX):
            continue

        cvals = [x[3] for x in labs]
        c_hi = _quantile(cvals, float(DESCRIPTOR_KEEP_C_QHI))
        labs2 = [x for x in labs if x[3] <= c_hi]
        if len(labs2) >= int(DESCRIPTOR_MIN_UNIQUE_HEX):
            labs = labs2

        chosen = _lab_medoid_hex([x[0] for x in labs])
        if chosen is None:
            continue
        hx, meta = chosen

        derived[t] = {
            "n_phrase_keys": int(nphr),
            "n_hex": int(len(set(hexes))),
            "n_sources": int(len(seen_src)),
            "single_confidence": int(min(len(seen_src), len(set(hexes)))),
            "anchor": hx,
            "L_anchor": float(meta["L"]),
            "C_anchor": float(meta["C"]),
            "role": "descriptor",
        }

    return derived


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
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
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

            entries.append(LexiconEntry(key=key, name=name.strip(), hex=nh, source=f"colors-name-lists:{lk}"))

    if include_webcolors:
        entries.extend(load_webcolors_css3_entries())

    # deterministic order
    entries_sorted = sorted(entries, key=lambda e: (e.key, e.source, e.name, e.hex))

    key_index = _build_key_index(entries_sorted)

    phrase_counts = _count_phrase_keys_for_bases(key_index)
    prefix_counts = _count_prefix_keys_for_tokens(key_index)

    out: Dict[str, Dict[str, Any]] = {}
    for k, info in key_index.items():
        if k in DENY_KEYS:
            continue
        chosen = _choose_anchor_for_key(k, info)
        if chosen is None:
            continue
        out[k] = {
            "name": chosen["name"],
            "hex": chosen["hex"],
            "anchor_policy": chosen["anchor_policy"],
            "n_hex": chosen["n_hex"],
            "n_sources": chosen["n_sources"],
            "L_anchor": float(chosen["L_anchor"]),
            "C_anchor": float(chosen["C_anchor"]),
            "single_confidence": int(min(int(chosen["n_sources"]), int(chosen["n_hex"]))),
            "n_phrase_keys": int(phrase_counts.get(k, 0)),
        }

    # Stabilize mono-token bases (generic)
    _stabilize_base_anchors(out, key_index=key_index)

    # NEW: remove modifier-like singletons (prevents "dark" resolving as a base colors)
    dropped_mod_singletons = _drop_modifier_like_singletons(out, phrase_counts=phrase_counts, prefix_counts=prefix_counts)

    # Descriptor derivation -> report only (do not inject into out)
    derived_descriptors = _derive_descriptors_report_only(key_index=key_index)

    report: Dict[str, Any] = {
        "meta": {
            "n_entries_raw": len(entries_sorted),
            "n_keys": len(out),
            "lists": list(list_keys),
            "include_webcolors": bool(include_webcolors),
            "dropped_modifier_singletons_n": int(len(dropped_mod_singletons)),
            "dropped_modifier_singletons_thresholds": {
                "min_phrase_keys": MOD_SINGLE_MIN_PHRASE_KEYS,
                "min_prefix_keys": MOD_SINGLE_MIN_PREFIX_KEYS,
                "min_prefix_frac": MOD_SINGLE_MIN_PREFIX_FRAC,
                "max_single_confidence": MOD_SINGLE_MAX_SINGLE_CONF,
                "max_n_sources": MOD_SINGLE_MAX_N_SOURCES,
                "max_n_hex": MOD_SINGLE_MAX_N_HEX,
            },
        },
        "anchors": {},
        "derived_descriptors": derived_descriptors,
        "dropped_modifier_singletons": dropped_mod_singletons,
    }

    for k, v in out.items():
        hx = v.get("hex")
        lab = _hex_to_lab(hx) if isinstance(hx, str) else None
        if lab is None:
            continue
        report["anchors"][k] = {
            "n_hex": v.get("n_hex"),
            "n_sources": v.get("n_sources"),
            "n_phrase_keys": v.get("n_phrase_keys"),
            "single_confidence": v.get("single_confidence"),
            "anchor": hx,
            "L_anchor": float(lab[0]),
            "C_anchor": float(_lab_chroma(lab)),
            "anchor_policy": v.get("anchor_policy"),
        }

    return out, report


def write_lexicon_json(path: Path, lex: Dict[str, Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    ordered = {k: lex[k] for k in sorted(lex.keys())}
    with path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


def write_report_json(path: Path, report: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build extended colors lexicon JSON (colors-name-lists + optional webcolors).")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT), help=f"Output JSON path (default: {DEFAULT_OUT})")
    p.add_argument("--report-out", type=str, default=str(DEFAULT_REPORT), help=f"Report JSON path (default: {DEFAULT_REPORT})")
    p.add_argument("--report", action="store_true", help="Write report JSON with anchor stats.")
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
    report_path = Path(args.report_out)
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
            raise RuntimeError("No lists available from colors-name-lists blob.")
        default = ["wikipedia", "xkcd", "ntc", "ral"]
        list_keys = default if all(k in available for k in default) else available[:1]
        print(f"[build_color_lexicon] No --lists provided. Using: {list_keys}")
    else:
        list_keys = list(args.lists)

    lex, report = build_lexicon(
        http,
        list_keys=list_keys,
        include_webcolors=bool(args.include_webcolors),
        force_fetch=bool(args.force_fetch),
        debug=bool(args.debug_payload),
    )

    write_lexicon_json(out_path, lex)
    print(f"[build_color_lexicon] Wrote {len(lex)} entries -> {out_path}")
    print(f"[build_color_lexicon] Cache dir: {cache_dir}")

    if args.report:
        write_report_json(report_path, report)
        print(f"[build_color_lexicon] Wrote report -> {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
