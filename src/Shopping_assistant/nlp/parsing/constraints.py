# src/Shopping_assistant/nlp/parsing/constraints.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

from Shopping_assistant.nlp.schema import Axis, Constraint, Direction, Polarity, Strength
from Shopping_assistant.nlp.axes.predictor import predict_axis

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Token
else:
    Language = Any  # type: ignore
    Doc = Any       # type: ignore
    Token = Any     # type: ignore


# ---------------------------------------------------------------------
# Axis mapping: embedding-first + dynamic fallback via axis families
# ---------------------------------------------------------------------

_AXIS_LEXICON: Dict[str, Axis] = {}  # keep empty by default

_NEG_LEFT = {"not", "no", "nothing"}
_DEG_LEFT = {"too", "more", "less"}  # comparatives / caps

_INTENSIFIERS = {
    "very", "really", "so", "extremely", "quite", "super", "highly", "pretty", "rather",
}

_COLOR_DOMAIN_NOUNS = {
    "color", "shade", "tone", "tint", "hue", "lipstick", "gloss", "balm", "liner", "stain",
}

_AXIS_FAMILY: Dict[Axis, str] = {
    Axis.BRIGHTNESS: "LIGHTNESS",
    Axis.DEPTH: "LIGHTNESS",
    Axis.SATURATION: "CHROMA",
    Axis.VIBRANCY: "CHROMA",
    Axis.CLARITY: "SURFACE",
}

_FAMILY_AXES: Dict[str, set[Axis]] = {}
for ax, fam in _AXIS_FAMILY.items():
    _FAMILY_AXES.setdefault(fam, set()).add(ax)


def _normalize_model_name(model_name: str) -> str:
    if "/" in (model_name or ""):
        return model_name
    return f"sentence-transformers/{model_name}"


def _norm_text_piece(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"n't", "nt"}:
        return "not"
    return s


def _clean_axis_core(tok: Token) -> str:
    return _norm_text_piece(tok.lemma_)


def _iter_axis_queries(tok: Token) -> List[str]:
    lemma = _clean_axis_core(tok)
    out: List[str] = []
    if lemma:
        out.append(lemma)
        out.append(f"{lemma} color")

    seen = set()
    uniq: List[str] = []
    for q in out:
        q = " ".join((q or "").split())
        if q and q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


def _axis_from_text(
    text: str,
    *,
    mapper_model: str,
    mapper_threshold: float,
    mapper_min_margin: float,
    axis_source: str,
    debug: bool = False,
) -> Tuple[Optional[Axis], Dict[str, Any], Any]:
    pred = predict_axis(
        text,
        context="",
        model_name=_normalize_model_name(mapper_model),
        min_sim=float(mapper_threshold),
        min_margin=float(mapper_min_margin),
        debug=bool(debug),
    )

    meta: Dict[str, Any] = {
        "axis_source": axis_source,
        "axis_score": float(getattr(pred, "confidence", 0.0)),
        "axis_margin": float(getattr(pred, "margin", 0.0)),
        "axis": pred.axis.value if getattr(pred, "axis", None) is not None else None,
        "axis_query": text,
    }

    # If predictor returns an axis, it is a STRICT pass by definition.
    if getattr(pred, "axis", None) is not None:
        meta["axis_gate"] = "STRICT"

    if debug:
        meta["axis_debug"] = getattr(pred, "meta", None)
    return pred.axis, meta, pred


def _ranked_axes_from_pred(pred: Any) -> List[Tuple[Axis, float]]:
    meta = getattr(pred, "meta", None)
    if not isinstance(meta, dict):
        return []

    ranked = meta.get("ranked") or meta.get("ranked_top5") or meta.get("ranked_top3")
    if not ranked:
        return []

    out: List[Tuple[Axis, float]] = []
    for a, s in ranked:
        try:
            ax = Axis(a) if isinstance(a, str) else a
            out.append((ax, float(s)))
        except Exception:
            continue
    return out


def _best_axis_from_pred(pred: Any) -> Optional[Axis]:
    ax = getattr(pred, "axis", None)
    if ax is not None:
        return ax

    meta = getattr(pred, "meta", None)
    if isinstance(meta, dict):
        ba = meta.get("best_axis")
        if isinstance(ba, str):
            try:
                return Axis(ba)
            except Exception:
                return None
    return None


def _context_is_colorish(tok: Token, *, window: int = 5) -> bool:
    head = getattr(tok, "head", None)
    if head is not None and head.pos_ in {"NOUN", "PROPN"} and _norm_text_piece(head.lemma_) in _COLOR_DOMAIN_NOUNS:
        return True

    doc = tok.doc
    for j in range(max(0, tok.i - window), min(len(doc), tok.i + window + 1)):
        t = doc[j]
        if t.pos_ in {"NOUN", "PROPN"} and _norm_text_piece(t.lemma_) in _COLOR_DOMAIN_NOUNS:
            return True
    return False


def _family(ax: Axis) -> str:
    return _AXIS_FAMILY.get(ax, "OTHER")


def _best_lightness_from_ranked(ranked: List[Tuple[Axis, float]]) -> Optional[Tuple[Axis, float]]:
    cand = [(a, s) for a, s in ranked if _family(a) == "LIGHTNESS"]
    if not cand:
        return None
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[0]


def _family_fallback(
    pred: Any,
    *,
    min_sim: float,
    min_margin: float,
    color_context: bool,
) -> Tuple[Optional[Axis], Dict[str, Any]]:
    conf = float(getattr(pred, "confidence", 0.0))
    margin = float(getattr(pred, "margin", 0.0))
    best_ax = _best_axis_from_pred(pred)

    meta: Dict[str, Any] = {}

    if best_ax is None:
        return None, meta

    if conf < float(min_sim):
        meta["axis_gate"] = "FAIL_MIN_SIM"
        return None, meta

    if margin >= float(min_margin):
        meta["axis_gate"] = "STRICT"
        return best_ax, meta

    ranked = _ranked_axes_from_pred(pred)
    if len(ranked) < 2:
        meta["axis_gate"] = "FAIL_NO_RANKED"
        return None, meta

    ax1, s1 = ranked[0]
    ax2, s2 = ranked[1]
    fam1 = _family(ax1)
    fam2 = _family(ax2)

    family_margin_floor = 0.02 if color_context else 0.0
    if margin < family_margin_floor:
        meta.update(
            {
                "axis_gate": "FAIL_MARGIN_TOO_FLAT",
                "axis_family_top1": fam1,
                "axis_family_top2": fam2,
                "axis_top2": [(ax1.value, float(s1)), (ax2.value, float(s2))],
                "axis_color_context": bool(color_context),
            }
        )
        return None, meta

    if fam1 == fam2 and fam1 in {"LIGHTNESS", "CHROMA"}:
        chosen_ax = ax1
        chosen_reason = "FAMILY"

        if fam1 == "CHROMA":
            light = _best_lightness_from_ranked(ranked[:3])
            if light is not None:
                axL, sL = light
                if float(sL) >= float(s1) - 0.03:
                    chosen_ax = axL
                    chosen_reason = "FAMILY_BIAS_LIGHTNESS"

        meta.update(
            {
                "axis_gate": chosen_reason,
                "axis_family": _family(chosen_ax),
                "axis_family_top1": fam1,
                "axis_family_top2": fam2,
                "axis_ambiguous": True,
                "axis_top2": [(ax1.value, float(s1)), (ax2.value, float(s2))],
                "axis_margin_used": float(margin),
                "axis_conf_used": float(conf),
                "axis_color_context": bool(color_context),
            }
        )
        return chosen_ax, meta

    meta.update(
        {
            "axis_gate": "FAIL_FAMILY",
            "axis_family_top1": fam1,
            "axis_family_top2": fam2,
            "axis_top2": [(ax1.value, float(s1)), (ax2.value, float(s2))],
            "axis_color_context": bool(color_context),
        }
    )
    return None, meta


def _axis_from_token(
    tok: Token,
    *,
    mapper_model: str = "all-MiniLM-L6-v2",
    mapper_threshold: float = 0.35,
    mapper_min_margin: float = 0.08,
) -> Tuple[Optional[Axis], Dict[str, Any]]:
    lemma = _norm_text_piece(tok.lemma_)
    base_meta: Dict[str, Any] = {"tok_lemma": lemma, "tok_pos": tok.pos_, "tok_text": tok.text}

    last_meta2: Dict[str, Any] = {}
    last_fb_meta: Dict[str, Any] = {}

    for q in _iter_axis_queries(tok):
        axis, meta2, pred = _axis_from_text(
            q,
            mapper_model=mapper_model,
            mapper_threshold=mapper_threshold,
            mapper_min_margin=mapper_min_margin,
            axis_source="embed",
            debug=True,
        )
        last_meta2 = dict(meta2)  # keep last attempt meta (no accumulation)
        last_fb_meta = {}

        # 1) strict pass
        if axis is not None:
            meta = {
                **base_meta,
                **meta2,  # meta matches the winning query
                "axis_query_strategy": "multi_try_first_pass",
                "axis_gate": meta2.get("axis_gate", "STRICT"),
            }
            return axis, meta

        # 2) family fallback (non-strict) based on THIS pred + THIS query
        best_ax = _best_axis_from_pred(pred)
        if best_ax is not None:
            color_ctx = _context_is_colorish(tok)
            ax_fb, fb_meta = _family_fallback(
                pred,
                min_sim=mapper_threshold,
                min_margin=mapper_min_margin,
                color_context=color_ctx,
            )
            last_fb_meta = dict(fb_meta)

            if ax_fb is not None:
                meta = {
                    **base_meta,
                    **meta2,     # meta matches the query that produced pred
                    **fb_meta,   # meta matches the fallback decision for that pred
                    "axis_query_strategy": "multi_try_family_fallback",
                    "axis": ax_fb.value,
                }
                return ax_fb, meta

    # 3) lexicon fallback
    axis2 = _AXIS_LEXICON.get(lemma)
    if axis2 is not None:
        meta = {
            **base_meta,
            "axis_source": "lexicon_fallback",
            "axis_score": 1.0,
            "axis_margin": 1.0,
            "axis": axis2.value,
            "axis_query": lemma,
            "axis_gate": "LEXICON",
            "axis_query_strategy": "lexicon_fallback",
        }
        return axis2, meta

    # 4) all failed: return ONLY last attempt meta (no mixed fields)
    meta = {
        **base_meta,
        **last_meta2,
        **last_fb_meta,
        "axis_query_strategy": "multi_try_all_failed",
    }
    return None, meta


# ---------------------------------------------------------------------
# Robust context signals
# ---------------------------------------------------------------------

def _has_negation(tok: Token) -> bool:
    return any(ch.dep_ == "neg" for ch in tok.children)


def _has_negation_anywhere(tok: Token, *, max_hops: int = 3, window: int = 3) -> bool:
    if _has_negation(tok):
        return True

    cur = tok
    for _ in range(max_hops):
        head = getattr(cur, "head", None)
        if head is None or head == cur:
            break
        if any(ch.dep_ == "neg" for ch in head.children):
            return True
        cur = head

    doc = tok.doc
    for j in range(1, window + 1):
        i = tok.i - j
        if i < 0:
            break
        if doc[i].lemma_.lower() in _NEG_LEFT or _norm_text_piece(doc[i].text) == "not":
            return True

    return False


def _degree_adverbs(tok: Token) -> List[Token]:
    out = [ch for ch in tok.children if ch.dep_ == "advmod" and ch.pos_ == "ADV"]

    doc = tok.doc
    for j in (tok.i - 1, tok.i - 2, tok.i - 3):
        if j >= 0:
            t = doc[j]
            lem = t.lemma_.lower()
            if t.pos_ == "ADV" and (lem in _DEG_LEFT or lem in _INTENSIFIERS):
                out.append(t)

    out = sorted({t.i: t for t in out}.values(), key=lambda t: t.i)
    return out


def _has_degree_anywhere(tok: Token) -> bool:
    return any(adv.lemma_.lower() in (_DEG_LEFT | _INTENSIFIERS) for adv in _degree_adverbs(tok))


def _degree_strength(tok: Token) -> Strength:
    deg = _degree_adverbs(tok)

    if not deg:
        return Strength.STRONG if _has_negation_anywhere(tok) else Strength.MED

    if len(deg) >= 2:
        return Strength.STRONG

    lemma = deg[0].lemma_.lower()
    if lemma in {"more", "less"}:
        return Strength.MED
    return Strength.STRONG


def _direction_from_context(tok: Token) -> Direction:
    direction = Direction.RAISE
    has_too = False

    for adv in _degree_adverbs(tok):
        lemma = adv.lemma_.lower()
        if lemma == "less":
            direction = Direction.LOWER
        elif lemma == "more":
            direction = Direction.RAISE
        elif lemma == "too":
            direction = Direction.LOWER
            has_too = True

    if _has_negation_anywhere(tok) and not has_too:
        direction = Direction.LOWER if direction == Direction.RAISE else Direction.RAISE

    return direction


def _evidence(tok: Token) -> tuple[str, int, int]:
    toks = [tok]

    for ch in tok.children:
        if ch.dep_ == "neg":
            toks.append(ch)
        elif ch.dep_ == "advmod" and ch.pos_ == "ADV":
            toks.append(ch)

    for adv in _degree_adverbs(tok):
        toks.append(adv)

    doc = tok.doc
    for base in list(toks):
        j = base.i - 1
        if j >= 0 and (doc[j].lemma_.lower() in _NEG_LEFT or _norm_text_piece(doc[j].text) == "not"):
            toks.append(doc[j])

    toks = sorted({t.i: t for t in toks}.values(), key=lambda t: t.i)

    start = toks[0].idx
    last = toks[-1]
    end = last.idx + len(last.text)

    evidence = " ".join(t.text for t in toks)
    return evidence, int(start), int(end)


def _is_adjectival_noun(tok: Token) -> bool:
    if tok.pos_ != "NOUN":
        return False
    if tok.is_stop or tok.is_punct or tok.is_space:
        return False
    if len(tok.text) <= 2:
        return False
    if tok.dep_ in {"amod", "acomp", "attr"}:
        return True
    if any(
        ch.dep_ == "advmod"
        and ch.lemma_.lower() in (_DEG_LEFT | _INTENSIFIERS | {"too"})
        for ch in tok.children
    ):
        return True
    return False


def _is_constraint_candidate(tok: Token) -> bool:
    if tok.is_stop or tok.is_punct or tok.is_space:
        return False
    if len(tok.text) <= 2:
        return False

    if tok.pos_ in {"VERB", "AUX"}:
        return False

    if tok.pos_ == "ADJ":
        return True
    if _is_adjectival_noun(tok):
        return True

    if _has_negation(tok) and tok.pos_ in {"ADJ", "ADV", "NOUN"}:
        return True
    if any(ch.dep_ == "advmod" and ch.lemma_.lower() in (_DEG_LEFT | _INTENSIFIERS) for ch in tok.children):
        return True

    return False


def _head_is_domain_noun(tok: Token) -> bool:
    head = getattr(tok, "head", None)
    if head is None:
        return False
    if head.pos_ not in {"NOUN", "PROPN"}:
        return False
    return _norm_text_piece(head.lemma_) in _COLOR_DOMAIN_NOUNS


# ---------------------------------------------------------------------
# Portfolio-facing hygiene: quality tag + dedup/merge
# ---------------------------------------------------------------------

def _quality_label(*, axis_gate: str, conf: float, margin: float) -> str:
    gate = (axis_gate or "").upper()
    if gate in {"STRICT", "LEXICON"} and conf >= 0.50 and margin >= 0.10:
        return "high"
    if gate in {"STRICT", "LEXICON"}:
        return "med"
    if conf >= 0.45 and margin >= 0.05:
        return "med"
    return "low"


def _strength_rank(s: Strength) -> int:
    # defensive: if schema changes, keep stable ordering
    if s == Strength.STRONG:
        return 3
    if s == Strength.MED:
        return 2
    if s == Strength.WEAK:
        return 1
    return 0


def _merge_constraints(constraints: List[Constraint]) -> List[Constraint]:
    """
    Merge duplicates for portfolio clarity.
    Key: (axis, direction, clause_id)
    - confidence: max
    - strength: max
    - evidence: unique concatenation
    - meta: keep meta from the best-confidence item + add evidence_parts
    """
    buckets: Dict[Tuple[Axis, Direction, int], List[Constraint]] = {}
    for c in constraints:
        buckets.setdefault((c.axis, c.direction, int(c.clause_id)), []).append(c)

    out: List[Constraint] = []
    for key, items in buckets.items():
        if len(items) == 1:
            c0 = items[0]
            meta = dict(c0.meta or {})
            meta.setdefault("evidence_parts", [c0.evidence] if c0.evidence else [])
            out.append(
                Constraint(
                    axis=c0.axis,
                    direction=c0.direction,
                    strength=c0.strength,
                    evidence=c0.evidence,
                    clause_id=c0.clause_id,
                    confidence=c0.confidence,
                    scope=c0.scope,
                    meta=meta,
                )
            )
            continue

        # choose "best" by confidence, then by strength
        items_sorted = sorted(items, key=lambda c: (float(c.confidence), _strength_rank(c.strength)), reverse=True)
        best = items_sorted[0]

        evidences: List[str] = []
        for c in items:
            ev = (c.evidence or "").strip()
            if ev and ev not in evidences:
                evidences.append(ev)

        merged_evidence = " / ".join(evidences) if evidences else best.evidence
        merged_conf = max(float(c.confidence) for c in items)
        merged_strength = max((c.strength for c in items), key=_strength_rank)

        meta = dict(best.meta or {})
        meta["evidence_parts"] = evidences
        meta["merged_count"] = len(items)
        meta["confidence_merged_max"] = merged_conf

        out.append(
            Constraint(
                axis=best.axis,
                direction=best.direction,
                strength=merged_strength,
                evidence=merged_evidence,
                clause_id=best.clause_id,
                confidence=merged_conf,
                scope=best.scope,
                meta=meta,
            )
        )

    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def extract_constraints_from_doc(
    doc: Doc,
    *,
    clause_id: int,
    clause_polarity: Polarity = Polarity.UNKNOWN,
    blocked_lemmas: Optional[set[str]] = None,
    mapper_model: str = "all-MiniLM-L6-v2",
    mapper_threshold: float = 0.35,
    mapper_min_margin: float = 0.08,
) -> List[Constraint]:
    out: List[Constraint] = []

    for tok in doc:
        if not _is_constraint_candidate(tok):
            continue

        lemma = tok.lemma_.lower()
        if blocked_lemmas and lemma in blocked_lemmas:
            continue

        axis, axis_meta = _axis_from_token(
            tok,
            mapper_model=mapper_model,
            mapper_threshold=mapper_threshold,
            mapper_min_margin=mapper_min_margin,
        )
        if axis is None:
            continue

        axis_score = float(axis_meta.get("axis_score", 0.0))
        axis_margin = float(axis_meta.get("axis_margin", 0.0))
        axis_gate = str(axis_meta.get("axis_gate", "STRICT"))
        fam = str(axis_meta.get("axis_family") or _family(axis))

        has_signal = bool(_has_negation_anywhere(tok) or _has_degree_anywhere(tok))
        head_domain = _head_is_domain_noun(tok)

        # -----------------------------------------------------------------
        # Kill ultra-ambiguous non-STRICT decisions early
        # (below ~0.03 margin, FAMILY picks are usually noise)
        # -----------------------------------------------------------------
        # allow LIGHTNESS family constraints when there is explicit signal (neg/degree),
        # even if margin is low (dark/bright frequently have tight margins)
        if axis_gate not in {"STRICT", "LEXICON"} and axis_margin < 0.03:
            has_signal = bool(_has_negation_anywhere(tok) or _has_degree_anywhere(tok))
            fam = str(axis_meta.get("axis_family") or _family(axis))
            if not (has_signal and fam == "LIGHTNESS"):
                continue

        # 1) Bare adjectives (likely colors) => drop unless extremely strong evidence
        if not has_signal:
            if not (axis_score >= 0.55 and axis_margin >= 0.10):
                continue

        # 2) Family/non-STRICT gating:
        #    - allow LIGHTNESS constraints with signal even with low margin (dark/bright-ish cases)
        #    - allow CHROMA constraints with signal when NOT modifying a domain noun
        #      (fixes: "nothing too strong", while still dropping "brown lipstick")
        if axis_gate not in {"STRICT", "LEXICON"}:
            if has_signal and fam == "LIGHTNESS":
                margin_floor = 0.02
                score_floor = 0.43
            elif has_signal and (fam == "CHROMA") and (not head_domain):
                margin_floor = 0.03
                score_floor = 0.45
            else:
                margin_floor = 0.05
                score_floor = 0.55

            if axis_margin < margin_floor or axis_score < score_floor:
                continue

        has_too = any(adv.lemma_.lower() == "too" for adv in _degree_adverbs(tok))
        if _has_negation_anywhere(tok) and not has_too:
            direction = Direction.LOWER
            strength = Strength.STRONG
        else:
            direction = _direction_from_context(tok)
            strength = _degree_strength(tok)

        evidence, ev_start, ev_end = _evidence(tok)

        quality = _quality_label(axis_gate=axis_gate, conf=axis_score, margin=axis_margin)

        out.append(
            Constraint(
                axis=axis,
                direction=direction,
                strength=strength,
                evidence=evidence,
                clause_id=clause_id,
                confidence=axis_score,
                scope=None,
                meta={
                    **axis_meta,
                    "tok": tok.text,
                    "negated": _has_negation_anywhere(tok),
                    "children": [(c.text, c.dep_, c.pos_, c.lemma_) for c in tok.children],
                    "direction": direction.value,
                    "strength": strength.value,
                    "evidence_char_start": ev_start,
                    "evidence_char_end": ev_end,
                    "clause_polarity": clause_polarity.value,
                    "has_signal": bool(has_signal),
                    "axis_family_effective": fam,
                    "head_is_domain_noun": bool(head_domain),
                    "quality": quality,
                },
            )
        )

    # Portfolio hygiene: merge duplicates (e.g., repeated "not neon" sentences)
    return _merge_constraints(out)


def extract_constraints_from_clause_text(
    clause_text: str,
    *,
    clause_id: int,
    clause_polarity: Polarity = Polarity.UNKNOWN,
    blocked_lemmas: Optional[set[str]] = None,
    nlp: Language,
    mapper_model: str = "all-MiniLM-L6-v2",
    mapper_threshold: float = 0.35,
    mapper_min_margin: float = 0.08,
) -> List[Constraint]:
    doc = nlp(clause_text)
    return extract_constraints_from_doc(
        doc,
        clause_id=clause_id,
        clause_polarity=clause_polarity,
        blocked_lemmas=blocked_lemmas,
        mapper_model=mapper_model,
        mapper_threshold=mapper_threshold,
        mapper_min_margin=mapper_min_margin,
    )


def extract_constraints(
    clauses: Iterable[Tuple[int, str]],
    *,
    clause_polarities: Optional[Dict[int, Polarity]] = None,
    nlp: Language,
    mapper_model: str = "all-MiniLM-L6-v2",
    mapper_threshold: float = 0.35,
    mapper_min_margin: float = 0.08,
) -> List[Constraint]:
    out: List[Constraint] = []
    for cid, text in clauses:
        doc = nlp(text)
        pol = Polarity.UNKNOWN if clause_polarities is None else clause_polarities.get(cid, Polarity.UNKNOWN)
        out.extend(
            extract_constraints_from_doc(
                doc,
                clause_id=cid,
                clause_polarity=pol,
                mapper_model=mapper_model,
                mapper_threshold=mapper_threshold,
                mapper_min_margin=mapper_min_margin,
            )
        )
    # extract_constraints_from_doc already merges per-doc; this is just a safe global merge
    return _merge_constraints(out)


__all__ = [
    "extract_constraints_from_doc",
    "extract_constraints_from_clause_text",
    "extract_constraints",
]
