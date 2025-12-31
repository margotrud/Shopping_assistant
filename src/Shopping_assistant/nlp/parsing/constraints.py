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
# Axis mapping: small optional lexicon + transformer fallback
# ---------------------------------------------------------------------

_AXIS_LEXICON: Dict[str, Axis] = {
    # brightness/lightness
    "bright": Axis.BRIGHTNESS,
    "light": Axis.BRIGHTNESS,

    # depth (often "dark/deep" language)
    "deep": Axis.DEPTH,
    "dark": Axis.DEPTH,

    # saturation/chroma
    "muted": Axis.SATURATION,
    "soft": Axis.SATURATION,
    "subtle": Axis.SATURATION,
    "saturated": Axis.SATURATION,

    # vibrancy/neonness
    "neon": Axis.VIBRANCY,
    "vibrant": Axis.VIBRANCY,

    # clarity
    "crisp": Axis.CLARITY,
    "clean": Axis.CLARITY,
    "muddy": Axis.CLARITY,
    "dull": Axis.CLARITY,
}


def _normalize_model_name(model_name: str) -> str:
    if "/" in (model_name or ""):
        return model_name
    return f"sentence-transformers/{model_name}"


def _axis_from_token(
    tok: Token,
    *,
    mapper_model: str = "all-MiniLM-L6-v2",
    mapper_threshold: float = 0.35,
) -> Tuple[Optional[Axis], Dict[str, Any]]:
    """
    Does:
        Map a descriptive token to one of the axes using:
        - small precision lexicon
        - semantic embedding predictor fallback
    """
    lemma = tok.lemma_.lower()
    meta: Dict[str, Any] = {"tok_lemma": lemma, "tok_pos": tok.pos_, "tok_text": tok.text}

    axis = _AXIS_LEXICON.get(lemma)
    if axis is not None:
        meta.update({"axis_source": "lexicon", "axis_score": 1.0, "axis": axis.value})
        return axis, meta

    pred = predict_axis(
        lemma,
        context="",
        model_name=_normalize_model_name(mapper_model),
        min_sim=float(mapper_threshold),
        min_margin=0.08,
        debug=False,
    )

    if pred.axis is None:
        meta.update({"axis_source": "embed", "axis_score": float(pred.confidence), "axis": None})
        return None, meta

    meta.update(
        {
            "axis_source": "embed",
            "axis_score": float(pred.confidence),
            "axis": pred.axis.value,
        }
    )
    return pred.axis, meta


# ---------------------------------------------------------------------
# Dependency-based signals
# ---------------------------------------------------------------------

def _has_negation(tok: Token) -> bool:
    return any(ch.dep_ == "neg" for ch in tok.children)


def _degree_adverbs(tok: Token) -> List[Token]:
    return [ch for ch in tok.children if ch.dep_ == "advmod" and ch.pos_ == "ADV"]


def _degree_strength(tok: Token) -> Strength:
    deg = _degree_adverbs(tok)
    if not deg:
        return Strength.STRONG if _has_negation(tok) else Strength.MED
    if len(deg) >= 2:
        return Strength.STRONG
    if deg[0].lemma_.lower() == "too":
        return Strength.STRONG
    return Strength.MED


def _direction_from_context(tok: Token) -> Direction:
    direction = Direction.RAISE
    has_too = False

    for ch in tok.children:
        if ch.dep_ == "advmod":
            lemma = ch.lemma_.lower()
            if lemma == "less":
                direction = Direction.LOWER
            elif lemma == "more":
                direction = Direction.RAISE
            elif lemma == "too":
                direction = Direction.LOWER
                has_too = True

    if _has_negation(tok) and not has_too:
        direction = Direction.LOWER if direction == Direction.RAISE else Direction.RAISE

    return direction


def _evidence(tok: Token) -> tuple[str, int, int]:
    keep_adv = {"too", "more", "less"}

    toks = [tok]

    advmods: List[Token] = []
    for ch in tok.children:
        if ch.dep_ == "neg":
            toks.append(ch)
        elif ch.dep_ == "advmod" and ch.pos_ == "ADV" and ch.lemma_.lower() in keep_adv:
            toks.append(ch)
            advmods.append(ch)

    doc = tok.doc
    for adv in advmods:
        j = adv.i - 1
        if j >= 0:
            left = doc[j]
            if left.lemma_.lower() == "not":
                toks.append(left)

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
    if any(ch.dep_ == "advmod" and ch.lemma_.lower() in {"too", "very", "so"} for ch in tok.children):
        return True
    return False


def _axis_from_negated_fallback(
    tok: Token,
    *,
    mapper_model: str,
    mapper_threshold: float,
) -> Tuple[Optional[Axis], Dict[str, Any]]:
    lemma = tok.lemma_.lower()
    pred = predict_axis(
        lemma,
        context="",
        model_name=_normalize_model_name(mapper_model),
        min_sim=float(max(0.20, mapper_threshold - 0.10)),
        min_margin=0.05,
        debug=False,
    )
    meta: Dict[str, Any] = {
        "tok_lemma": lemma,
        "tok_pos": tok.pos_,
        "tok_text": tok.text,
        "axis_source": "embed_neg_fallback",
        "axis_score": float(pred.confidence),
        "axis": pred.axis.value if pred.axis is not None else None,
    }
    return pred.axis, meta


def _is_constraint_candidate(tok: Token) -> bool:
    """
    Does:
        Decide whether a token is eligible for constraint extraction.

    Important:
        Accept elliptical constructions like "not neon / not flashy" even if spaCy tags
        the adjective-like token as NOUN/PROPN (common with punctuation like '/').
    """
    if tok.is_stop or tok.is_punct or tok.is_space:
        return False
    if len(tok.text) <= 2:
        return False

    if tok.pos_ == "ADJ":
        return True
    if _is_adjectival_noun(tok):
        return True

    # NEW: accept if it has explicit negation or degree modifier (covers POS mis-tags)
    if _has_negation(tok):
        return True
    if any(ch.dep_ == "advmod" and ch.lemma_.lower() in {"too", "more", "less"} for ch in tok.children):
        return True

    return False


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
) -> List[Constraint]:
    """
    Does:
        Extract constraints using spaCy dependency structure + transformer axis mapping fallback.
    """
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
        )

        # support simple negation "not X" dynamically
        if axis is None and _has_negation(tok):
            axis2, axis_meta2 = _axis_from_negated_fallback(
                tok,
                mapper_model=mapper_model,
                mapper_threshold=mapper_threshold,
            )
            if axis2 is not None:
                axis = axis2
                axis_meta = axis_meta2

        if axis is None:
            continue

        # For negated adjectives without explicit degree ("not neon", "not flashy"),
        # force a cap behavior: LOWER + STRONG. (No inversion ambiguity.)
        if _has_negation(tok) and not any(ch.dep_ == "advmod" and ch.lemma_.lower() == "too" for ch in tok.children):
            direction = Direction.LOWER
            strength = Strength.STRONG
        else:
            direction = _direction_from_context(tok)
            strength = _degree_strength(tok)

        evidence, ev_start, ev_end = _evidence(tok)

        out.append(
            Constraint(
                axis=axis,
                direction=direction,
                strength=strength,
                evidence=evidence,
                clause_id=clause_id,
                confidence=float(axis_meta.get("axis_score", 0.0)),
                scope=None,
                meta={
                    **axis_meta,
                    "tok": tok.text,
                    "negated": _has_negation(tok),
                    "children": [(c.text, c.dep_, c.pos_, c.lemma_) for c in tok.children],
                    "direction": direction.value,
                    "strength": strength.value,
                    "evidence_char_start": ev_start,
                    "evidence_char_end": ev_end,
                    "clause_polarity": clause_polarity.value,
                },
            )
        )

    return out


def extract_constraints_from_clause_text(
    clause_text: str,
    *,
    clause_id: int,
    clause_polarity: Polarity = Polarity.UNKNOWN,
    blocked_lemmas: Optional[set[str]] = None,
    nlp: Language,
    mapper_model: str = "all-MiniLM-L6-v2",
    mapper_threshold: float = 0.35,
) -> List[Constraint]:
    """
    Important:
        `nlp` must be provided (single-load spaCy). This function does NOT load spaCy.
    """
    doc = nlp(clause_text)
    return extract_constraints_from_doc(
        doc,
        clause_id=clause_id,
        clause_polarity=clause_polarity,
        blocked_lemmas=blocked_lemmas,
        mapper_model=mapper_model,
        mapper_threshold=mapper_threshold,
    )


def extract_constraints(
    clauses: Iterable[Tuple[int, str]],
    *,
    clause_polarities: Optional[Dict[int, Polarity]] = None,
    nlp: Language,
    mapper_model: str = "all-MiniLM-L6-v2",
    mapper_threshold: float = 0.35,
) -> List[Constraint]:
    """
    Does:
        Extract constraints from multiple (clause_id, clause_text) pairs.
    """
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
            )
        )
    return out


__all__ = [
    "extract_constraints_from_doc",
    "extract_constraints_from_clause_text",
    "extract_constraints",
]
