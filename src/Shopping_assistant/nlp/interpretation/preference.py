# src/Shopping_assistant/nlp/preference.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from Shopping_assistant.nlp.resolve import resolve_preference
from Shopping_assistant.nlp.parsing.clauses import split_clauses, ClauseSplitConfig
from Shopping_assistant.nlp.llm.analyze_clauses import build_world_alias_index, extract_mentions_free
from Shopping_assistant.nlp.parsing.polarity import make_free_polarity_fn, infer_polarity_for_mentions
from Shopping_assistant.nlp.parsing.constraints import extract_constraints_from_clause_text
from Shopping_assistant.nlp.schema import (
    NLPResult,
    Clause,
    Mention,
    MentionKind,
    Polarity,
    Span,
    Constraint,
    Strength,
)
from Shopping_assistant.utils.optional_deps import require

__all__ = [
    "interpret_nlp",
    "interpret_preference_text",
    "build_preference_from_nlp",
    "build_preference_from_text",
]

_DEFAULT_CLAUSE_CFG: ClauseSplitConfig = {
    "split_pos": ["CCONJ", "SCONJ"],
    "split_dep": ["cc", "mark", "conj", "advcl", "ccomp", "xcomp", "acl", "relcl"],
    "split_punct": [",", ";", "—", "-", "(", ")", ":", "."],
    "keep_len_min": 3,
    "keep_len_max": None,
}


def _clause_global_offset(cl: Clause) -> int:
    chunk = cl.meta.get("chunk") or {}
    start = chunk.get("start_char")
    return int(start) if isinstance(start, int) else 0


def _mention_span_from_doc(doc: Any, tok_start: Any, tok_len: Any) -> Span:
    """
    Does:
        Convert (tok_start, tok_len) into a clause-local Span, with end exclusive.
        Returns Span(0,0) if inputs are missing/invalid.
    """
    if not isinstance(tok_start, int) or not isinstance(tok_len, int) or tok_len <= 0:
        return Span(0, 0)
    if len(doc) <= 0:
        return Span(0, 0)

    start_i = max(0, min(tok_start, len(doc) - 1))
    end_excl = max(start_i + 1, min(tok_start + tok_len, len(doc)))
    start = int(doc[start_i].idx)
    end_tok = doc[end_excl - 1]
    end = int(end_tok.idx + len(end_tok.text))
    return Span(start, end)


def _to_polarity(v: Any) -> Polarity:
    try:
        s = str(v).strip().lower()
    except Exception:
        return Polarity.UNKNOWN

    if s in {"like", "pos", "positive", "want", "prefer", "liked"} or v == "LIKE":
        return Polarity.LIKE
    if s in {"dislike", "neg", "negative", "avoid", "not", "disliked"} or v == "DISLIKE":
        return Polarity.DISLIKE
    if s in {"neutral", "none"}:
        return Polarity.NEUTRAL
    return Polarity.UNKNOWN


def interpret_nlp(
    text: str,
    *,
    spacy_model: str = "en_core_web_sm",
    clause_config: Optional[ClauseSplitConfig] = None,
    include_xkcd: bool = True,
    debug: bool = False,
) -> NLPResult:
    """Does: Parse text into clauses, color mentions (global spans), and typed constraints."""
    cfg = clause_config or _DEFAULT_CLAUSE_CFG
    clauses = split_clauses(text, spacy_model=spacy_model, config=cfg, debug=debug)

    spacy = require("spacy", extra="spacy", purpose="NLP clause analysis")
    nlp = spacy.load(spacy_model)

    color_index = build_world_alias_index(include_xkcd=include_xkcd)

    # IMPORTANT: do NOT pass spaCy nlp object here
    pol_fn = make_free_polarity_fn(debug=debug)

    all_mentions: List[Mention] = []
    all_constraints: List[Constraint] = []

    diagnostics: Dict[str, Any] = {}
    if debug:
        diagnostics["clauses"] = [
            {
                "clause_id": c.clause_id,
                "text": c.text,
                "elliptical_neg": c.elliptical_neg,
                "reason": c.reason,
                "meta": c.meta,
            }
            for c in clauses
        ]
        diagnostics["mentions"] = []
        diagnostics["constraints"] = []

    for cl in clauses:
        clause_text = (cl.text or "").strip()
        if not clause_text:
            continue

        clause_offset = _clause_global_offset(cl)
        doc = nlp(clause_text)

        # -----------------------------
        # Mentions (colors)
        # -----------------------------
        mention_dicts = extract_mentions_free(clause_text, color_index)

        mention_names = [
            str(m.get("name") or "").strip()
            for m in mention_dicts
            if isinstance(m, dict)
        ]
        mention_names = [m for m in mention_names if m]
        mention_set = set(mention_names)

        pol_map = infer_polarity_for_mentions(
            clause_text,
            mention_names,
            llm_polarity_fn=pol_fn,
            elliptical_neg=cl.elliptical_neg,
        )

        for m in mention_dicts:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or "").strip()
            if not name or name == "lipstick":
                continue

            span_local = _mention_span_from_doc(doc, m.get("tok_start"), m.get("tok_len"))
            span_global = Span(span_local.start + clause_offset, span_local.end + clause_offset)

            all_mentions.append(
                Mention(
                    span=span_global,
                    raw=str(m.get("alias") or name),
                    canonical=name,
                    kind=MentionKind.COLOR,
                    polarity=_to_polarity(pol_map.get(name) if name in mention_set else None),
                    clause_id=cl.clause_id,
                    confidence=float(m.get("confidence") or 1.0),
                    meta={"hue_deg": m.get("hue_deg"), "chunk": cl.meta.get("chunk")},
                )
            )

        # -----------------------------
        # Constraints (typed, canonical)
        # -----------------------------
        # ✅ CRITICAL: block color lemmas so they don't become axis constraints (saturation, etc.)
        cons = extract_constraints_from_clause_text(
            clause_text,
            clause_id=cl.clause_id,
            blocked_lemmas=mention_set,
            nlp=nlp,
            spacy_model=spacy_model,
        )

        for c in cons:
            meta = dict(c.meta or {})
            s = meta.get("evidence_char_start")
            e = meta.get("evidence_char_end")
            if isinstance(s, int) and isinstance(e, int):
                meta["evidence_global_start"] = s + clause_offset
                meta["evidence_global_end"] = e + clause_offset

            all_constraints.append(
                Constraint(
                    axis=c.axis,
                    direction=c.direction,
                    strength=c.strength,
                    evidence=c.evidence,
                    clause_id=c.clause_id,
                    confidence=c.confidence,
                    scope=c.scope,
                    meta=meta,
                )
            )

        if debug:
            diagnostics["mentions"].extend(
                [
                    {
                        "clause_id": m.clause_id,
                        "raw": m.raw,
                        "canonical": m.canonical,
                        "polarity": m.polarity.value,
                        "span": (m.span.start, m.span.end),
                        "meta": m.meta,
                    }
                    for m in all_mentions
                    if m.clause_id == cl.clause_id
                ]
            )
            diagnostics["constraints"].extend(
                [
                    {
                        "clause_id": c.clause_id,
                        "axis": c.axis.value,
                        "direction": c.direction.value,
                        "strength": c.strength.value,
                        "evidence": c.evidence,
                        "confidence": c.confidence,
                        "scope": c.scope,
                        "meta": c.meta,
                    }
                    for c in all_constraints
                    if c.clause_id == cl.clause_id
                ]
            )

    return NLPResult(
        text=text,
        clauses=tuple(clauses),
        mentions=tuple(all_mentions),
        constraints=tuple(all_constraints),
        diagnostics=diagnostics,
    )


def interpret_preference_text(
    text: str,
    *,
    spacy_model: str = "en_core_web_sm",
    clause_config: Optional[ClauseSplitConfig] = None,
    include_xkcd: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """Does: Convenience wrapper returning a JSON-friendly dict for apps/CLI."""
    res = interpret_nlp(
        text,
        spacy_model=spacy_model,
        clause_config=clause_config,
        include_xkcd=include_xkcd,
        debug=debug,
    )
    out: Dict[str, Any] = {
        "text": res.text,
        "clauses": [
            {
                "clause_id": c.clause_id,
                "text": c.text,
                "elliptical_neg": c.elliptical_neg,
                "reason": c.reason,
                "meta": c.meta,
            }
            for c in res.clauses
        ],
        "mentions": [
            {
                "clause_id": m.clause_id,
                "raw": m.raw,
                "canonical": m.canonical,
                "kind": m.kind.value,
                "polarity": m.polarity.value,
                "span": (m.span.start, m.span.end),
                "confidence": m.confidence,
                "meta": m.meta,
            }
            for m in res.mentions
        ],
        "constraints": [
            {
                "clause_id": c.clause_id,
                "axis": c.axis.value,
                "direction": c.direction.value,
                "strength": c.strength.value,
                "evidence": c.evidence,
                "confidence": c.confidence,
                "scope": c.scope,
                "meta": c.meta,
            }
            for c in res.constraints
        ],
    }
    if debug:
        out["_DEBUG"] = res.diagnostics
    return out


def _dedupe_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def build_preference_from_nlp(nlp_res: NLPResult) -> Dict[str, Any]:
    """
    Does:
        Convert NLPResult into a scoring-friendly preference payload:
        likes/dislikes + hard/soft constraints (stable order, deduped),
        using deterministic resolution for scoping.
    """
    resolved = resolve_preference(nlp_res)

    likes = _dedupe_preserve_order([t.mention.canonical for t in resolved.liked if t.mention.canonical])
    dislikes = _dedupe_preserve_order([t.mention.canonical for t in resolved.disliked if t.mention.canonical])

    # Collect constraints from attached + globals
    all_constraints: List[Constraint] = []
    for t in resolved.liked:
        all_constraints.extend(list(t.constraints))
    for t in resolved.disliked:
        all_constraints.extend(list(t.constraints))
    all_constraints.extend([gc.constraint for gc in resolved.global_constraints])

    hard: List[Constraint] = []
    soft: List[Constraint] = []
    for c in all_constraints:
        if c.strength == Strength.STRONG:
            hard.append(c)
        else:
            soft.append(c)

    return {
        "likes": likes,
        "dislikes": dislikes,
        "hard_constraints": hard,
        "soft_constraints": soft,
        # utile pour debug (tu peux retirer plus tard)
        "nlp_diagnostics": resolved.diagnostics,
    }


def build_preference_from_text(
    text: str,
    *,
    spacy_model: str = "en_core_web_sm",
    clause_config: Optional[ClauseSplitConfig] = None,
    include_xkcd: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """Does: interpret_nlp(text) then build_preference_from_nlp(result)."""
    nlp_res = interpret_nlp(
        text,
        spacy_model=spacy_model,
        clause_config=clause_config,
        include_xkcd=include_xkcd,
        debug=debug,
    )
    return build_preference_from_nlp(nlp_res)
