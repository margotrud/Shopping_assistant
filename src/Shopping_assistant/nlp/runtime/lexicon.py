# src/Shopping_assistant/nlp/runtime/lexicon.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from Shopping_assistant.utils.optional_deps import require  # NEW (dynamic stopwords, not hardcoded denylist)

SPACE_RE = re.compile(r"\s+")
HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


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


@lru_cache(maxsize=1)
def _stop_words_en() -> set[str]:
    """
    Dynamic guardrail: rely on spaCy STOP_WORDS instead of a hardcoded denylist.
    """
    spacy = require("spacy", extra="spacy", purpose="STOP_WORDS for color lexicon guardrails.")
    return set(spacy.lang.en.stop_words.STOP_WORDS)


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
    """
    existing = set(idx.keys())

    for k in list(existing):
        info = idx.get(k)
        if not info:
            continue

        for v in _token_variants(k):
            if v == k:
                continue
            if v in existing:
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


def _fuzzy_ratio(a: str, b: str) -> float:
    return 100.0 * SequenceMatcher(None, a, b).ratio()


# -----------------------------
# Semantic resolution helpers
# -----------------------------
def _project_root() -> Path:
    # pythonProject/src/Shopping_assistant/nlp/runtime/lexicon.py -> pythonProject
    return Path(__file__).resolve().parents[4]


def _lexicon_cache_dir() -> Path:
    return _project_root() / "data" / "cache" / "color_lexicon"


def _safe_model_id(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "__", model_name.strip())


def _default_semantic_model() -> str:
    return os.environ.get(
        "SA_COLOR_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )


@lru_cache(maxsize=2)
def _st_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Semantic resolution requires sentence-transformers. "
            "Install: pip install sentence-transformers"
        ) from e
    return SentenceTransformer(model_name)


def _load_or_build_key_embeddings(keys: List[str], model_name: str) -> np.ndarray:
    cache_dir = _lexicon_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    p = cache_dir / f"lexicon_emb__{_safe_model_id(model_name)}.npz"
    if p.exists():
        data = np.load(p, allow_pickle=True)
        cached_keys = data["keys"].tolist()
        if cached_keys == keys:
            return data["emb"].astype(np.float32)

    model = _st_model(model_name)
    emb = model.encode(keys, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)

    np.savez_compressed(p, keys=np.asarray(keys, dtype=object), emb=emb)
    return emb


def _semantic_topk(
    query: str,
    keys: List[str],
    key_emb: np.ndarray,
    model_name: str,
    topk: int,
) -> List[tuple[str, float]]:
    model = _st_model(model_name)
    q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    q_emb = np.asarray(q_emb, dtype=np.float32)[0]

    sims = key_emb @ q_emb  # cosine (embeddings are normalized)
    k = max(1, int(topk))

    if k >= len(keys):
        idxs = np.argsort(-sims)
    else:
        idxs = np.argpartition(-sims, kth=k - 1)[:k]
        idxs = idxs[np.argsort(-sims[idxs])]

    return [(keys[i], float(sims[i])) for i in idxs]


@dataclass(frozen=True, slots=True)
class ResolvedColor:
    alias: str
    name: str
    hex: str
    source: str
    score: float


@dataclass(frozen=True)
class ColorLexicon:
    """
    Loads data/nlp/color_lexicon.json and exposes:
      - raw_index: alias -> info
      - mention scan using token-variant normalization (peachy/reddish/brownish)

    API-compat:
      - resolve() returns ResolvedColor objects (attribute access),
        as expected by callers in nlp/llm/analyze_clauses.py.
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
            if not kk:
                continue

            hx = _norm_hex(str(v.get("hex") or ""))
            if not hx:
                continue

            idx[kk] = {
                "name": str(v.get("name") or kk),
                "hex": hx,
                "source": str(v.get("source") or ""),
            }

        _derive_missing_base_aliases(idx)
        return cls(raw_index=idx)

    def resolve(
        self,
        query: str,
        *,
        topk: int = 1,
        fuzzy_cutoff: float = 75.0,
        use_semantic: bool = False,
    ) -> List[ResolvedColor]:
        """
        Resolve a free-text candidate to lexicon entries.

        Returns:
          List[ResolvedColor] with attribute access (.hex, .name, ...).

        Notes:
          - fuzzy matching uses SequenceMatcher as a cheap runtime fallback.
          - semantic fallback (sentence-transformers) is optional and cached.
          - semantic is guarded to avoid resolving constraints / multi-token phrases.
        """
        use_sem = bool(use_semantic)

        q = _norm_key(query)
        if not q:
            return []

        # Dynamic guardrail: reject stopwords (prevents sure->surf, etc.)
        if q in _stop_words_en():
            return []

        # Dynamic guardrail: reject axis-like tokens (prevents bright->bright red, etc.)
        try:
            from Shopping_assistant.nlp.axes.predictor import predict_axis  # local import to avoid cycles
            pred = predict_axis(q, debug=False)
            if getattr(pred, "axis", None) is not None:
                return []
        except Exception:
            pass

        info = self.raw_index.get(q)
        if info:
            return [
                ResolvedColor(
                    alias=q,
                    name=str(info.get("name", q)),
                    hex=str(info.get("hex")),
                    source=str(info.get("source", "")),
                    score=100.0,
                )
            ][: max(1, int(topk))]

        # single-token variant preference
        if " " not in q:
            k2 = _prefer_variant(q, self.raw_index)
            if k2 and k2 != q:
                info2 = self.raw_index.get(k2)
                if info2:
                    return [
                        ResolvedColor(
                            alias=k2,
                            name=str(info2.get("name", k2)),
                            hex=str(info2.get("hex")),
                            source=str(info2.get("source", "")),
                            score=99.0,
                        )
                    ][: max(1, int(topk))]

        # -----------------------------
        # Fuzzy fallback (guarded)
        # -----------------------------
        single = (" " not in q)

        # 1) disable fuzzy for short single tokens (avoids sure->surf, loud->cloud, lip->lilac)
        min_len = int(os.environ.get("SA_COLOR_FUZZY_SINGLE_MIN_LEN", "6"))
        use_fuzzy = not (single and len(q) < min_len)

        # 2) stronger cutoff for single-token fuzzy
        cutoff = float(fuzzy_cutoff)
        if single:
            cutoff = float(os.environ.get("SA_COLOR_FUZZY_CUTOFF_SINGLE", "92.0"))

        if use_fuzzy:
            best_k: Optional[str] = None
            best_s = 0.0

            for k in self.raw_index.keys():
                # Never match single-token -> multi-token via fuzzy (bright->bright red)
                if single and " " in k:
                    continue

                if single and abs(len(k) - len(q)) > 6:
                    continue

                s = _fuzzy_ratio(q, k)
                if s > best_s:
                    best_s = s
                    best_k = k

            if best_k is not None and best_s >= cutoff:
                info3 = self.raw_index[best_k]
                return [
                    ResolvedColor(
                        alias=best_k,
                        name=str(info3.get("name", best_k)),
                        hex=str(info3.get("hex")),
                        source=str(info3.get("source", "")),
                        score=float(best_s),
                    )
                ][: max(1, int(topk))]

        # semantic fallback (keep existing guardrails; still dynamic)
        if use_sem:
            # 1) Never semantic-resolve multi-token phrases (constraints, style, negations, etc.)
            if " " in q:
                return []

            # 2) Never semantic-resolve tokens that look like axis labels
            try:
                from Shopping_assistant.nlp.axes.predictor import predict_axis  # local import to avoid cycles
                pred = predict_axis(q, debug=False)
                if getattr(pred, "axis", None) is not None:
                    return []
            except Exception:
                pass

            model_name = _default_semantic_model()
            keys = list(self.raw_index.keys())
            if keys:
                key_emb = _load_or_build_key_embeddings(keys, model_name)
                cand = _semantic_topk(q, keys, key_emb, model_name, topk=max(1, int(topk)))
                if not cand:
                    return []

                # Require strong top-1 similarity to accept semantic resolution
                top1_min = float(os.environ.get("SA_COLOR_SEMANTIC_TOP1_MIN", "0.62"))
                _top1_k, top1_sim = cand[0]
                if top1_sim < top1_min:
                    return []

                sem_cutoff = float(os.environ.get("SA_COLOR_SEMANTIC_CUTOFF", "0.55"))

                out: List[ResolvedColor] = []
                for k_sem, sim in cand:
                    if sim < sem_cutoff:
                        continue
                    info4 = self.raw_index.get(k_sem)
                    if not info4:
                        continue
                    out.append(
                        ResolvedColor(
                            alias=k_sem,
                            name=str(info4.get("name", k_sem)),
                            hex=str(info4.get("hex")),
                            source=str(info4.get("source", "")) + ":semantic",
                            score=100.0 * sim,
                        )
                    )
                return out[: max(1, int(topk))]

        return []

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


def load_default_lexicon() -> ColorLexicon:
    return ColorLexicon.load()
