# src/Shopping_assistant/nlp/preference.py
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Any, Dict, List, Optional

from Shopping_assistant.nlp.resolve import resolve_preference
from Shopping_assistant.nlp.resolve.conflicts import resolve_symbolic_conflicts
from Shopping_assistant.nlp.parsing.clauses import split_clauses, ClauseSplitConfig
from Shopping_assistant.nlp.llm.analyze_clauses import build_world_alias_index, extract_mentions_free
from Shopping_assistant.nlp.parsing.polarity import (
    make_free_polarity_fn,
    infer_polarity_for_mentions,
    decide_clause_polarity,
)
from Shopping_assistant.nlp.parsing.constraints import extract_constraints_from_clause_text
from Shopping_assistant.nlp.schema import (
    NLPResult,
    Clause,
    Mention,
    MentionKind,
    Polarity,
    Span,
    Constraint,
)
from Shopping_assistant.utils.optional_deps import require

__all__ = [
    "interpret_nlp",
    "interpret_preference_text",
    "build_preference_from_text",
    "build_preference_from_nlp",
]


_DEFAULT_CLAUSE_CFG: ClauseSplitConfig = {
    "split_pos": ["CCONJ", "SCONJ"],
    "split_dep": ["cc", "mark", "conj", "advcl", "ccomp", "xcomp", "acl", "relcl"],
    "split_punct": [",", ";", ":"],
    "keep_len_min": 3,
    "keep_len_max": None,
}


@lru_cache(maxsize=2)
def _load_spacy(model: str):
    spacy = require(
        "spacy",
        extra="spacy",
        purpose="Needed for interpret_nlp() (single-load spaCy runtime).",
    )
    return spacy.load(model)


def _to_polarity(x: Optional[str]) -> Polarity:
    if not isinstance(x, str):
        return Polarity.UNKNOWN
    v = x.strip().upper()
    if v == "LIKE":
        return Polarity.LIKE
    if v == "DISLIKE":
        return Polarity.DISLIKE
    return Polarity.UNKNOWN


def _clause_global_offset(cl: Clause) -> int:
    ch = (cl.meta or {}).get("chunk") if isinstance(cl.meta, dict) else None
    if isinstance(ch, dict):
        s = ch.get("start_char")
        if isinstance(s, int):
            return s
    return 0


def _mention_span_from_doc(doc: Any, tok_start: Any, tok_len: Any) -> Span:
    if not isinstance(tok_start, int) or not isinstance(tok_len, int) or tok_len <= 0:
        return Span(0, 0)
    if tok_start < 0 or tok_start >= len(doc):
        return Span(0, 0)
    end_i = min(len(doc), tok_start + tok_len)
    span = doc[tok_start:end_i]
    if not hasattr(span, "start_char") or not hasattr(span, "end_char"):
        return Span(0, 0)
    return Span(int(span.start_char), int(span.end_char))


def interpret_nlp(
    text: str,
    *,
    spacy_model: str = "en_core_web_sm",
    clause_config: Optional[ClauseSplitConfig] = None,
    include_xkcd: bool = True,
    debug: bool = False,
) -> NLPResult:
    # ------------------------------------------------------------
    # Single-load spaCy runtime
    # ------------------------------------------------------------
    nlp = _load_spacy(spacy_model)

    cfg = clause_config or _DEFAULT_CLAUSE_CFG
    clauses_in = split_clauses(
        text,
        nlp=nlp,
        spacy_model=spacy_model,
        config=cfg,
        debug=debug,
    )

    # World alias index (colors etc.)
    color_index = build_world_alias_index(include_xkcd=include_xkcd)

    # Polarity backend (offline ST)
    pol_fn = make_free_polarity_fn()

    all_mentions: List[Mention] = []
    all_constraints: List[Constraint] = []
    clauses_out: List[Clause] = []

    diagnostics: Dict[str, Any] = {}
    if debug:
        diagnostics["clauses"] = [
            {
                "clause_id": c.clause_id,
                "text": c.text,
                "polarity": c.polarity,
                "elliptical_neg": c.elliptical_neg,
                "reason": c.reason,
                "meta": c.meta,
            }
            for c in clauses_in
        ]
        diagnostics["mentions"] = []
        diagnostics["constraints"] = []

    for cl in clauses_in:
        clause_text = (cl.text or "").strip()
        if not clause_text:
            clauses_out.append(cl)
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

        cl_polarity = decide_clause_polarity(
            pol_map,
            elliptical_neg=cl.elliptical_neg,
        )

        clauses_out.append(replace(cl, polarity=cl_polarity))

        for m in mention_dicts:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or "").strip()
            if not name or name == "lipstick":
                continue

            span_local = _mention_span_from_doc(
                doc, m.get("tok_start"), m.get("tok_len")
            )
            span_global = Span(
                span_local.start + clause_offset,
                span_local.end + clause_offset,
            )

            mention_obj = Mention(
                span=span_global,
                raw=str(m.get("alias") or name),
                canonical=name,
                kind=MentionKind.COLOR,
                polarity=_to_polarity(pol_map.get(name) if name in mention_set else None),
                clause_id=cl.clause_id,
                confidence=float(m.get("confidence") or 1.0),
                meta={"hue_deg": m.get("hue_deg"), "chunk": cl.meta.get("chunk")},
            )

            if mention_obj.polarity == Polarity.UNKNOWN:
                mention_obj = replace(mention_obj, polarity=cl_polarity)

            all_mentions.append(mention_obj)

        # -----------------------------
        # Constraints
        # -----------------------------
        cons = extract_constraints_from_clause_text(
            clause_text,
            clause_id=cl.clause_id,
            clause_polarity=cl_polarity,
            blocked_lemmas=mention_set,
            nlp=nlp,
        )

        for c in cons:
            meta = dict(c.meta or {})
            s = meta.get("evidence_char_start")
            e = meta.get("evidence_char_end")
            if isinstance(s, int) and isinstance(e, int):
                meta["evidence_global_start"] = s + clause_offset
                meta["evidence_global_end"] = e + clause_offset
            all_constraints.append(replace(c, meta=meta))

        if debug:
            diagnostics["mentions"].extend(
                [
                    {
                        "clause_id": m.clause_id,
                        "canonical": m.canonical,
                        "raw": m.raw,
                        "polarity": m.polarity,
                        "confidence": m.confidence,
                        "span": {"start": m.span.start, "end": m.span.end},
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
                        "axis": c.axis,
                        "direction": c.direction,
                        "strength": c.strength,
                        "evidence": c.evidence,
                        "confidence": c.confidence,
                        "scope": c.scope,
                        "meta": c.meta,
                    }
                    for c in all_constraints
                    if c.clause_id == cl.clause_id
                ]
            )

    constraints_final, conflicts_diag = resolve_symbolic_conflicts(tuple(all_constraints))
    diagnostics = dict(diagnostics or {})
    diagnostics["conflicts"] = conflicts_diag

    return NLPResult(
        text=text,
        clauses=tuple(clauses_out) if clauses_out else tuple(clauses_in),
        mentions=tuple(all_mentions),
        constraints=constraints_final,
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
    res = interpret_nlp(
        text,
        spacy_model=spacy_model,
        clause_config=clause_config,
        include_xkcd=include_xkcd,
        debug=debug,
    )
    return {
        "text": res.text,
        "clauses": [c.__dict__ for c in res.clauses],
        "mentions": [m.__dict__ for m in res.mentions],
        "constraints": [c.__dict__ for c in res.constraints],
        "diagnostics": res.diagnostics,
    }


def build_preference_from_nlp(nlp_res: NLPResult) -> Dict[str, Any]:
    return resolve_preference(nlp_res)


def build_preference_from_text(
    text: str,
    *,
    spacy_model: str = "en_core_web_sm",
    clause_config: Optional[ClauseSplitConfig] = None,
    include_xkcd: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    nlp_res = interpret_nlp(
        text,
        spacy_model=spacy_model,
        clause_config=clause_config,
        include_xkcd=include_xkcd,
        debug=debug,
    )
    return build_preference_from_nlp(nlp_res)
