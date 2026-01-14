# src/Shopping_assistant/nlp/runtime/lexicon.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

SPACE_RE = re.compile(r"\s+")
HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


# Safety: runtime deny too (defense in depth)
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

# Back-compat alias used in a few internal snippets (keep both names)
DENY_TOKENS = DENY_KEYS


def _norm_text(s: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in s).split())


def _norm_key(name: str) -> str:
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


def _token_variants(t: str) -> List[str]:
    """
    Runtime-only morphological normalization.

    IMPORTANT:
    - This may generate suffix-stripped candidates for matching.
    - DO NOT materialize those candidates into the lexicon unless safe
      (see _derive_missing_base_aliases()).
    """
    t = t.strip().lower()
    if not t:
        return []
    out = [t]

    # reddish -> red, brownish -> brown
    if len(t) >= 5 and t.endswith("ish"):
        out.append(t[:-3])

    # peachy -> peach (BUT: berry -> berr is NOT a valid base; derive guard prevents materialization)
    if len(t) >= 4 and t.endswith("y"):
        out.append(t[:-1])

    # plural-s noise for color tokens (rare but cheap)
    if len(t) >= 4 and t.endswith("s"):
        out.append(t[:-1])

    # keep order, unique
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x and x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _derive_missing_base_aliases(idx: Dict[str, Dict[str, Any]]) -> None:
    """
    Add derived base aliases from existing keys (ish/y/s stripping),
    only when the base alias is missing, *and* when derivation is safe.

    Safety rule (critical):
      - Never create y-stripped bases when missing.
        This prevents noun truncation like "berry" -> "berr",
        "strawberry" -> "strawberr", etc.

    Example:
      - if "brownish" exists but "brown" doesn't, create "brown" -> same hex.
      - if "peachy" exists and "peach" does NOT exist, we do NOT auto-create "peach".
        (If you want "peach", it should exist as a real lexicon entry.)
    """
    existing = set(idx.keys())

    for k in list(existing):
        if k in DENY_KEYS:
            continue
        info = idx.get(k)
        if not info:
            continue

        for v in _token_variants(k):
            if v == k:
                continue
            if v in existing:
                continue
            if v in DENY_KEYS:
                continue

            # hard safety: never materialize y-stripped base (don't create new base from "...y")
            if k.endswith("y") and v == k[:-1]:
                continue

            idx[v] = {
                "name": str(v).title(),
                "hex": info["hex"],
                "source": f"{info.get('source', '')}:derived",
            }
            existing.add(v)


def _max_ngram_from_index(color_index: Dict[str, Dict[str, Any]]) -> int:
    if not color_index:
        return 1
    return min(max(len(a.split()) for a in color_index.keys()), 4)


def _prefer_variant(token: str, index: Dict[str, Dict[str, Any]]) -> str:
    """
    Choose the best matching variant for a token, preferring base forms
    (brown over brownish, peach over peachy) when both exist in the lexicon.

    Note:
      - This only prefers among forms that already exist in the lexicon.
      - It does not create new entries.
    """
    t = token.strip().lower()
    vars_ = _token_variants(t)

    present = [v for v in vars_ if v in index]
    if not present:
        return vars_[0] if vars_ else t

    # Prefer: not-original first, then shorter (base form tends to be shorter)
    present.sort(key=lambda v: (v == t, len(v)))
    return present[0]


@dataclass(frozen=True)
class ColorLexicon:
    """
    Loads data/nlp/color_lexicon.json and exposes:
      - raw_index: alias -> info
      - mention scan using token-variant normalization (peachy/reddish/brownish)
    """

    raw_index: Dict[str, Dict[str, Any]]

    @staticmethod
    def default_path() -> Path:
        # pythonProject/src/Shopping_assistant/nlp/runtime/lexicon.py -> pythonProject
        root = Path(__file__).resolve().parents[4]
        return root / "data" / "nlp" / "color_lexicon.json"

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ColorLexicon":
        p = path or cls.default_path()
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError(f"color_lexicon.json unexpected type: {type(data)}")

        idx: Dict[str, Dict[str, Any]] = {}
        for k, v in data.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                continue

            kk = _norm_key(k)
            if not kk or kk in DENY_KEYS:
                continue

            hx = _norm_hex(str(v.get("hex") or ""))
            if not hx:
                continue

            idx[kk] = {
                "name": str(v.get("name") or kk),
                "hex": hx,
                "source": str(v.get("source") or ""),
            }

        # Add missing base aliases dynamically (no manual mapping)
        # (y-stripping is blocked from materialization by _derive_missing_base_aliases)
        _derive_missing_base_aliases(idx)

        return cls(raw_index=idx)

    def extract_mentions(self, text: str) -> List[Dict[str, Any]]:
        toks = _norm_text(text).split()
        if not toks:
            return []

        max_n = _max_ngram_from_index(self.raw_index)
        hits: List[Dict[str, Any]] = []

        i = 0
        while i < len(toks):
            matched: Optional[Dict[str, Any]] = None

            token_cands = [
                _token_variants(toks[j]) for j in range(i, min(len(toks), i + max_n))
            ]

            for n in range(min(max_n, len(toks) - i), 0, -1):
                parts: List[str] = []
                ok = True
                for j in range(n):
                    cands = token_cands[j]
                    if not cands:
                        ok = False
                        break
                    parts.append(_prefer_variant(toks[i + j], self.raw_index))
                if not ok:
                    continue

                cand0 = " ".join(parts)
                info = self.raw_index.get(cand0)
                if info:
                    matched = {
                        "alias": cand0,
                        "name": info.get("name", cand0),
                        "tok_start": i,
                        "tok_len": n,
                    }
                    i += n
                    break

                # If not matched, allow variant substitution for the LAST token
                if n == 1:
                    best = _prefer_variant(toks[i], self.raw_index)
                    info2 = self.raw_index.get(best)
                    if info2:
                        matched = {
                            "alias": best,
                            "name": info2.get("name", best),
                            "tok_start": i,
                            "tok_len": 1,
                        }
                        i += 1
                        break
                else:
                    base_prefix = " ".join(parts[:-1])
                    alts_last = sorted(
                        token_cands[n - 1],
                        key=lambda s: (len(s), 0 if s != toks[i + n - 1] else 1),
                    )
                    for alt_last in alts_last:
                        cand = f"{base_prefix} {alt_last}".strip()
                        info2 = self.raw_index.get(cand)
                        if info2:
                            matched = {
                                "alias": cand,
                                "name": info2.get("name", cand),
                                "tok_start": i,
                                "tok_len": n,
                            }
                            i += n
                            break
                    if matched:
                        break

            if matched:
                hits.append(matched)
            else:
                i += 1

        # De-dup by name
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for h in hits:
            nm = str(h.get("name") or "")
            if nm and nm not in seen:
                uniq.append(h)
                seen.add(nm)
        return uniq


@lru_cache(maxsize=1)
def load_default_lexicon() -> ColorLexicon:
    lex = ColorLexicon.load()

    # Nude: add a single-word business alias (no multiword static combos)
    if "nude" not in lex.raw_index:
        # pick a neutral base that actually exists (minimal, deterministic)
        for base_key in ("beige", "tan"):
            if base_key in lex.raw_index:
                base = dict(lex.raw_index[base_key])  # clone
                base["name"] = "Nude"
                base["source"] = f'{base.get("source","")}:alias:nude'
                lex.raw_index["nude"] = base
                break

    return lex
