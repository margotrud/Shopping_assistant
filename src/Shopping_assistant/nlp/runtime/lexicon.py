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

from Shopping_assistant.nlp.runtime.spacy_runtime import load_spacy
from Shopping_assistant.utils.optional_deps import require

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
    Does:
        Return spaCy English stopwords (cached) via the project spaCy runtime loader.
    """
    try:
        nlp = load_spacy("en_core_web_sm")
        return set(getattr(nlp.Defaults, "stop_words", set()))
    except Exception:
        spacy = require("spacy", extra="spacy", purpose="STOP_WORDS fallback for colors lexicon guardrails.")
        return set(spacy.lang.en.stop_words.STOP_WORDS)


def _token_variants(t: str) -> List[str]:
    """
    Runtime-only morphological normalization.
    """
    t = t.strip().lower()
    if not t:
        return []
    out = [t]

    if len(t) >= 5 and t.endswith("ish"):
        out.append(t[:-3])

    if len(t) >= 4 and t.endswith("y"):
        out.append(t[:-1])

    # plural stripping: guarded (avoid glass->glas, dress->dres, etc.)
    if len(t) >= 5 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        out.append(t[:-1])

    seen = set()
    uniq: List[str] = []
    for x in out:
        if x and x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _prefer_variant(token: str, index: Dict[str, Dict[str, Any]]) -> str:
    """
    Choose the best matching variant for a token, preferring exact/base forms.
    """
    t = token.strip().lower()
    vars_ = _token_variants(t)

    present = [v for v in vars_ if v in index]
    if not present:
        return vars_[0] if vars_ else t

    # exact first, then shortest
    present.sort(key=lambda v: (0 if v == t else 1, len(v)))
    return present[0]


def _derive_missing_base_aliases(idx: Dict[str, Dict[str, Any]]) -> None:
    """
    Add derived base aliases from existing keys (ish/y/s stripping),
    only when the base alias is missing, *and* when derivation is safe.

    Safety rule (critical):
      - Never create y-stripped bases when missing.
        This prevents noun truncation like "berry" -> "berr".
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

            # never create y-stripped bases
            if k.endswith("y") and v == k[:-1]:
                continue

            idx[v] = {
                "name": str(v).title(),
                "hex": info["hex"],
                "source": f"{info.get('source', '')}:derived",
                "n_sources": int(info.get("n_sources", 0) or 0),
                "n_hex": int(info.get("n_hex", 0) or 0),
                "anchor_policy": info.get("anchor_policy"),
                "single_confidence": int(info.get("single_confidence", 0) or 0),
                "n_phrase_keys": int(info.get("n_phrase_keys", 0) or 0),
                "role": info.get("role"),
            }
            existing.add(v)


def _max_ngram_from_index(color_index: Dict[str, Dict[str, Any]]) -> int:
    if not color_index:
        return 1
    return min(max(len(a.split()) for a in color_index.keys()), 4)


def _is_descriptor(info: Dict[str, Any]) -> bool:
    role = str(info.get("role") or "").strip().lower()
    return role == "descriptor"


def _single_ok(info: Dict[str, Any]) -> bool:
    """
    Data-driven single-token guardrail for *non-exact* resolution (fuzzy/semantic).
    Uses (1) builder's single_confidence if present, else (2) fallback (n_sources,n_hex).
    Also blocks descriptor-like tokens (e.g. "nude") from resolving as a base colors.
    """
    if _is_descriptor(info):
        return False

    sc = int(info.get("single_confidence", 0) or 0)
    min_sc = int(os.environ.get("SA_COLOR_MIN_SINGLE_CONFIDENCE", "2"))
    if sc > 0:
        return sc >= min_sc

    min_sources = int(os.environ.get("SA_COLOR_MIN_SOURCES_SINGLE", "2"))
    min_hex = int(os.environ.get("SA_COLOR_MIN_HEX_SINGLE", "2"))

    ns = int(info.get("n_sources", 0) or 0)
    nh = int(info.get("n_hex", 0) or 0)
    return (ns >= min_sources) and (nh >= min_hex)


def _axis_override_ok(info: Dict[str, Any]) -> bool:
    """
    When axis predictor fires on a token that exists in the colors lexicon,
    allow the exact colors only if it has strong evidence of being a real colors/family.

    Data-driven (no static token lists):
      - single_confidence high OR phrase usage high OR sources/hex sufficient.
    """
    if _is_descriptor(info):
        return False

    sc = int(info.get("single_confidence", 0) or 0)
    nphr = int(info.get("n_phrase_keys", 0) or 0)
    ns = int(info.get("n_sources", 0) or 0)
    nh = int(info.get("n_hex", 0) or 0)

    min_sc = int(os.environ.get("SA_COLOR_AXIS_OVERRIDE_MIN_SC", "3"))
    min_nphr = int(os.environ.get("SA_COLOR_AXIS_OVERRIDE_MIN_PHRASE_KEYS", "5"))
    min_ns = int(os.environ.get("SA_COLOR_AXIS_OVERRIDE_MIN_SOURCES", "3"))
    min_nh = int(os.environ.get("SA_COLOR_AXIS_OVERRIDE_MIN_HEX", "2"))

    if sc >= min_sc:
        return True
    if nphr >= min_nphr:
        return True
    return (ns >= min_ns) and (nh >= min_nh)


def _fuzzy_ratio(a: str, b: str) -> float:
    return 100.0 * SequenceMatcher(None, a, b).ratio()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _lexicon_cache_dir() -> Path:
    return _project_root() / "data" / "cache" / "color_lexicon"


def _safe_model_id(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "__", model_name.strip())


def _default_semantic_model() -> str:
    return os.environ.get("SA_COLOR_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=2)
def _st_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Semantic resolution requires sentence-transformers. Install: pip install sentence-transformers"
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

    sims = key_emb @ q_emb
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
                "n_sources": int(v.get("n_sources", 0) or 0),
                "n_hex": int(v.get("n_hex", 0) or 0),
                "anchor_policy": v.get("anchor_policy"),
                "single_confidence": int(v.get("single_confidence", 0) or 0),
                "n_phrase_keys": int(v.get("n_phrase_keys", 0) or 0),
                "role": v.get("role"),
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
        allow_exact: bool = True,
    ) -> List[ResolvedColor]:
        """
        Resolve a free-text candidate to lexicon entries.

        Policy:
          - Axis predictor is consulted early for single tokens.
            If it fires, we return [] UNLESS the token has strong colors evidence in raw_index.
          - Exact hit: return it (except descriptors).
          - Variant hit: return it (except descriptors).
          - Fuzzy/Semantic: guarded via _single_ok (and descriptors blocked).
        """
        use_sem = bool(use_semantic)

        q = _norm_key(query)
        if not q:
            return []

        if q in _stop_words_en():
            return []

        # ------------------------------------------------------------------
        # Early axis guardrail for single tokens, with data-driven override.
        # Prevents "dark" resolving as a colors, without killing real colors
        # like "purple"/"beige" when predict_axis has false positives.
        # ------------------------------------------------------------------
        if " " not in q:
            try:
                from Shopping_assistant.nlp.axes.predictor import predict_axis  # local import to avoid cycles

                pred = predict_axis(q, debug=False)
                if getattr(pred, "axis", None) is not None:
                    info0 = self.raw_index.get(q)
                    if not info0:
                        return []
                    if not _axis_override_ok(info0):
                        return []
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Exact match (do NOT apply _single_ok here; only block descriptors).
        # ------------------------------------------------------------------
        if allow_exact:
            info = self.raw_index.get(q)
            if info:
                if (" " not in q) and _is_descriptor(info):
                    return []
                return [
                    ResolvedColor(
                        alias=q,
                        name=str(info.get("name", q)),
                        hex=str(info.get("hex")),
                        source=str(info.get("source") or "color_lexicon"),
                        score=100.0,
                    )
                ][: max(1, int(topk))]

        # ------------------------------------------------------------------
        # Variant match (morphological): also no consensus gate; only block descriptors.
        # ------------------------------------------------------------------
        if " " not in q:
            k2 = _prefer_variant(q, self.raw_index)
            if k2 and k2 != q:
                info2 = self.raw_index.get(k2)
                if info2:
                    if _is_descriptor(info2):
                        return []
                    return [
                        ResolvedColor(
                            alias=k2,
                            name=str(info2.get("name", k2)),
                            hex=str(info2.get("hex")),
                            source=str(info2.get("source") or "color_lexicon"),
                            score=99.0,
                        )
                    ][: max(1, int(topk))]

        # -----------------------------
        # Fuzzy fallback (guarded)
        # -----------------------------
        single = (" " not in q)

        min_len = int(os.environ.get("SA_COLOR_FUZZY_SINGLE_MIN_LEN", "6"))
        use_fuzzy = not (single and len(q) < min_len)

        cutoff = float(fuzzy_cutoff)
        if single:
            cutoff = float(os.environ.get("SA_COLOR_FUZZY_CUTOFF_SINGLE", "92.0"))

        if use_fuzzy:
            best_k: Optional[str] = None
            best_s = 0.0

            for k in self.raw_index.keys():
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
                if single and not _single_ok(info3):
                    return []
                return [
                    ResolvedColor(
                        alias=best_k,
                        name=str(info3.get("name", best_k)),
                        hex=str(info3.get("hex")),
                        source=str(info3.get("source") or "color_lexicon"),
                        score=float(best_s),
                    )
                ][: max(1, int(topk))]

        # semantic fallback
        if use_sem:
            if " " in q:
                return []

            model_name = _default_semantic_model()
            keys = sorted(self.raw_index.keys())  # stable cache key order
            if keys:
                key_emb = _load_or_build_key_embeddings(keys, model_name)
                cand = _semantic_topk(q, keys, key_emb, model_name, topk=max(1, int(topk)))
                if not cand:
                    return []

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
                    if not _single_ok(info4):
                        continue
                    out.append(
                        ResolvedColor(
                            alias=k_sem,
                            name=str(info4.get("name", k_sem)),
                            hex=str(info4.get("hex")),
                            source=str(info4.get("source") or "color_lexicon") + ":semantic",
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

            token_cands = [_token_variants(toks[j]) for j in range(i, min(len(toks), i + max_n))]

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
                    # block descriptors for single-token mentions (keeps "nude" out)
                    if n == 1 and _is_descriptor(info):
                        matched = None
                    else:
                        matched = {
                            "alias": cand0,
                            "name": info.get("name", cand0),
                            "tok_start": i,
                            "tok_len": n,
                        }
                        i += n
                        break

                if n == 1 and not matched:
                    best = _prefer_variant(toks[i], self.raw_index)
                    info2 = self.raw_index.get(best)
                    if info2 and not _is_descriptor(info2):
                        matched = {
                            "alias": best,
                            "name": info2.get("name", best),
                            "tok_start": i,
                            "tok_len": 1,
                        }
                        i += 1
                        break
                elif not matched and n > 1:
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
