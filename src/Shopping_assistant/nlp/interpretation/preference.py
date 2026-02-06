# src/Shopping_assistant/nlp/preference.py
"""Preference interpretation from text to a scoring-ready spec.

Does: Converts parsed mentions + constraints into a structured preference representation, including polarity,
resolved color mentions, and axis-level targets used by recommendation/scoring.
Public API: interpret_nlp() (and any other non-underscore entrypoints imported by reco/recommend and tests).
Inputs: raw text (or pre-parsed mentions/constraints) plus AssetBundle resources (lexicon, thresholds, conflicts).
Outputs: interpretation dict/objects including color tokens/anchors, axis targets, and ambiguity diagnostics.
Errors: raises ValueError for invalid arguments; may propagate asset/lexicon errors if resources are inconsistent.
"""

from __future__ import annotations

import re
from dataclasses import is_dataclass, replace
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set

from Shopping_assistant.nlp.axes.axis_family import resolve_axis_family
from Shopping_assistant.nlp.llm.analyze_clauses import build_world_alias_index, extract_mentions_free
from Shopping_assistant.nlp.parsing.clauses import ClauseSplitConfig, split_clauses
from Shopping_assistant.nlp.parsing.constraints import extract_constraints_from_clause_text
from Shopping_assistant.nlp.parsing.polarity import (
    decide_clause_polarity,
    infer_polarity_for_mentions,
    make_free_polarity_fn,
)
from Shopping_assistant.nlp.resolve.conflicts import resolve_symbolic_conflicts
from Shopping_assistant.nlp.resolve.constraint_normalizer import normalize_constraints
from Shopping_assistant.nlp.runtime.lexicon import load_default_lexicon
from Shopping_assistant.nlp.runtime.spacy_runtime import load_spacy
from Shopping_assistant.nlp.schema import (
    Clause,
    Constraint,
    Mention,
    MentionKind,
    NLPResult,
    Polarity,
    Span,
)

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

_TOKEN_SPLIT_RE = re.compile(r"[^\w]+")


def build_preference_from_nlp(nlp_res: NLPResult) -> Dict[str, Any]:
    """Does: convert NLPResult into a normalized preference dict.
    Used by: recommendation and scoring adapters.
    """
    # Local import to avoid circular import at module import-time.
    from Shopping_assistant.nlp.resolve import resolve_preference

    return resolve_preference(nlp_res)


@lru_cache(maxsize=2)
def _get_color_index(include_xkcd: bool) -> Dict[str, Dict[str, Any]]:
    """
    Does:
        Build the alias index used for mention extraction: default lexicon first, optional CSS/XKCD additive only.
    """
    idx = dict(load_default_lexicon().raw_index)

    if include_xkcd:
        world = build_world_alias_index(include_xkcd=True)
        for k, v in world.items():
            idx.setdefault(k, v)

    return idx


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
    """
    Does:
        Convert (tok_start, tok_len) into a local character span using the spaCy doc.
    """
    if not isinstance(tok_start, int) or not isinstance(tok_len, int) or tok_len <= 0:
        return Span(0, 0)
    if tok_start < 0 or tok_start >= len(doc):
        return Span(0, 0)
    end_i = min(len(doc), tok_start + tok_len)
    span = doc[tok_start:end_i]
    if not hasattr(span, "start_char") or not hasattr(span, "end_char"):
        return Span(0, 0)
    return Span(int(span.start_char), int(span.end_char))


def _mention_sort_key(m: Mention) -> tuple:
    return (
        int(m.clause_id),
        int(m.span.start),
        str(m.kind.value),
        str(m.canonical),
        str(m.raw),
    )


def _constraint_sort_key(c: Constraint) -> tuple:
    """
    Important:
        This must be None-safe because direction/axis can be Optional after conflict resolution/normalization.
    """
    meta = c.meta or {}
    gs = meta.get("evidence_global_start")
    ge = meta.get("evidence_global_end")
    gs_i = int(gs) if isinstance(gs, int) else 10**9
    ge_i = int(ge) if isinstance(ge, int) else 10**9

    ax = c.axis.value if getattr(c, "axis", None) is not None else ""
    dr = c.direction.value if getattr(c, "direction", None) is not None else ""
    st = c.strength.value if getattr(c, "strength", None) is not None else ""

    return (
        int(getattr(c, "clause_id", 0) or 0),
        str(ax),
        str(dr),
        str(st),
        gs_i,
        ge_i,
        str(getattr(c, "evidence", "") or ""),
    )


def _inject_axis_family(c: Constraint) -> Constraint:
    """
    Does:
        Inject axis_family_effective into constraint.meta via resolve_axis_family().
    """
    meta = dict(c.meta or {})
    family = resolve_axis_family({"axis": c.axis.value if c.axis else None, "meta": meta})
    if family:
        meta["axis_family_effective"] = family
    return replace(c, meta=meta)


def _blocked_lemmas_from_mention_dicts(mention_dicts: List[Dict[str, Any]]) -> Set[str]:
    """
    Does:
        Build a token blocklist from extracted mentions (raw alias + canonical name).
    """
    blocked: Set[str] = set()
    for m in mention_dicts:
        if not isinstance(m, dict):
            continue
        for field in (m.get("alias"), m.get("name")):
            s = (field or "")
            for tok in _TOKEN_SPLIT_RE.split(str(s).lower()):
                if tok:
                    blocked.add(tok)
    return blocked


def _to_jsonable(x: Any) -> Any:
    """
    Does:
        Convert dataclasses/slots/enums/tuples to JSON-serializable primitives for debug outputs.
    """
    from dataclasses import asdict
    from enum import Enum

    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Enum):
        return x.value
    if is_dataclass(x):
        return {k: _to_jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]
    if hasattr(x, "__slots__"):
        out = {}
        for k in getattr(x, "__slots__", []) or []:
            if isinstance(k, str) and hasattr(x, k):
                out[k] = _to_jsonable(getattr(x, k))
        if out:
            return out
    return str(x)


def _enum_value_or_none(x: Any) -> Any:
    """
    Does:
        Safely return Enum.value if Enum-like, else None (for Optional enums).
    """
    if x is None:
        return None
    v = getattr(x, "value", None)
    return v if v is not None else None


def interpret_nlp(
    text: str,
    *,
    spacy_model: str = "en_core_web_sm",
    clause_config: Optional[ClauseSplitConfig] = None,
    include_xkcd: bool = True,
    debug: bool = False,
) -> NLPResult:
    """
    Parse a free-text query into structured NLP signals used by the recommender.

    Does: split text into clauses, extract color mentions, infer polarity,
    extract and normalize symbolic constraints (axes, direction, strength).
    Returns: NLPResult containing clauses, mentions, constraints, and optional
    diagnostics/trace when debug=True.
    """
    nlp = load_spacy(spacy_model)

    cfg = clause_config or _DEFAULT_CLAUSE_CFG
    clauses_in = split_clauses(
        text,
        nlp=nlp,
        spacy_model=spacy_model,
        config=cfg,
        debug=debug,
    )

    color_index = _get_color_index(include_xkcd)
    pol_fn = make_free_polarity_fn()

    all_mentions: List[Mention] = []
    all_constraints: List[Constraint] = []
    clauses_out: List[Clause] = []

    diagnostics: Dict[str, Any] = {}
    trace: Optional[Dict[str, Any]] = None

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

        trace = {
            "input_text": text,
            "spacy_model": spacy_model,
            "include_xkcd": include_xkcd,
            "clauses": [],
            "mentions": [],
            "constraints_raw": [],
            "constraints_final": [],
            "conflicts": None,
        }

    for cl in clauses_in:
        clause_text = (cl.text or "").strip()
        if not clause_text:
            clauses_out.append(cl)
            continue

        clause_offset = _clause_global_offset(cl)
        doc = nlp(clause_text)

        mention_dicts = extract_mentions_free(clause_text, color_index, doc=doc)

        mention_names = [
            str(m.get("name") or "").strip()
            for m in mention_dicts
            if isinstance(m, dict)
        ]
        mention_names = [m for m in mention_names if m]
        mention_set = set(mention_names)

        blocked_lemmas = _blocked_lemmas_from_mention_dicts(
            [m for m in mention_dicts if isinstance(m, dict)]
        )

        pol_map = infer_polarity_for_mentions(
            clause_text,
            mention_names,
            llm_polarity_fn=pol_fn,
            elliptical_neg=cl.elliptical_neg,
        )

        cl_polarity = decide_clause_polarity(pol_map, elliptical_neg=cl.elliptical_neg)
        clauses_out.append(replace(cl, polarity=cl_polarity))

        if debug and isinstance(trace, dict):
            trace["clauses"].append(
                {
                    "clause_id": cl.clause_id,
                    "text": clause_text,
                    "elliptical_neg": bool(cl.elliptical_neg),
                    "polarity": cl_polarity.value,
                    "offset": int(clause_offset),
                }
            )

        for m in mention_dicts:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or "").strip()
            if not name or name == "lipstick":
                continue

            span_local = _mention_span_from_doc(doc, m.get("tok_start"), m.get("tok_len"))
            span_global = Span(span_local.start + clause_offset, span_local.end + clause_offset)

            mention_obj = Mention(
                span=span_global,
                raw=str(m.get("alias") or name),
                canonical=name,
                kind=MentionKind.COLOR,
                polarity=_to_polarity(pol_map.get(name) if name in mention_set else None),
                clause_id=cl.clause_id,
                confidence=float(m.get("confidence") or 1.0),
                meta={
                    "hex": m.get("hex"),
                    "hue_deg": m.get("hue_deg"),
                    "lab_L": m.get("lab_L"),
                    "lab_a": m.get("lab_a"),
                    "lab_b": m.get("lab_b"),
                    "chunk": (cl.meta or {}).get("chunk") if isinstance(cl.meta, dict) else None,
                },
            )

            if mention_obj.polarity == Polarity.UNKNOWN:
                mention_obj = replace(mention_obj, polarity=cl_polarity)

            all_mentions.append(mention_obj)

            if debug and isinstance(trace, dict):
                trace["mentions"].append(
                    {
                        "clause_id": mention_obj.clause_id,
                        "canonical": mention_obj.canonical,
                        "raw": mention_obj.raw,
                        "polarity": mention_obj.polarity.value,
                        "confidence": float(mention_obj.confidence),
                        "span": {
                            "start": int(mention_obj.span.start),
                            "end": int(mention_obj.span.end),
                        },
                        "meta": mention_obj.meta,
                    }
                )

        cons = extract_constraints_from_clause_text(
            clause_text,
            clause_id=cl.clause_id,
            clause_polarity=cl_polarity,
            blocked_lemmas=blocked_lemmas,
            nlp=nlp,
        )

        for c in cons:
            meta = dict(c.meta or {})
            s = meta.get("evidence_char_start")
            e = meta.get("evidence_char_end")
            if isinstance(s, int) and isinstance(e, int):
                meta["evidence_global_start"] = s + clause_offset
                meta["evidence_global_end"] = e + clause_offset

            c2 = replace(c, meta=meta)
            all_constraints.append(c2)

            if debug and isinstance(trace, dict):
                trace["constraints_raw"].append(
                    {
                        "clause_id": int(getattr(c2, "clause_id", 0) or 0),
                        "axis": _enum_value_or_none(getattr(c2, "axis", None)),
                        "direction": _enum_value_or_none(getattr(c2, "direction", None)),
                        "strength": _enum_value_or_none(getattr(c2, "strength", None)),
                        "confidence": float(getattr(c2, "confidence", 0.0) or 0.0),
                        "evidence": getattr(c2, "evidence", None),
                        "meta": getattr(c2, "meta", None),
                    }
                )

        if debug:
            diagnostics["mentions"].extend(
                [
                    {
                        "clause_id": m2.clause_id,
                        "canonical": m2.canonical,
                        "raw": m2.raw,
                        "polarity": m2.polarity,
                        "confidence": m2.confidence,
                        "span": {"start": m2.span.start, "end": m2.span.end},
                        "meta": m2.meta,
                    }
                    for m2 in all_mentions
                    if m2.clause_id == cl.clause_id
                ]
            )
            diagnostics["constraints"].extend(
                [
                    {
                        "clause_id": c3.clause_id,
                        "axis": c3.axis,
                        "direction": c3.direction,
                        "strength": c3.strength,
                        "evidence": c3.evidence,
                        "confidence": c3.confidence,
                        "scope": c3.scope,
                        "meta": c3.meta,
                    }
                    for c3 in all_constraints
                    if c3.clause_id == cl.clause_id
                ]
            )

    # Conflict resolution can legally yield constraints with optional fields (axis/direction canceled)
    constraints_final, conflicts_diag = resolve_symbolic_conflicts(tuple(all_constraints))

    # Hard filter: constraints without axis OR direction are not projectable and should not participate downstream.
    constraints_final = tuple(
        c
        for c in constraints_final
        if getattr(c, "axis", None) is not None and getattr(c, "direction", None) is not None
    )

    constraints_final = tuple(_inject_axis_family(c) for c in constraints_final)
    constraints_final = normalize_constraints(constraints_final, strict=debug)

    constraints_sorted = tuple(sorted(constraints_final, key=_constraint_sort_key))
    mentions_sorted = tuple(sorted(all_mentions, key=_mention_sort_key))

    diagnostics = dict(diagnostics or {})
    diagnostics["conflicts"] = conflicts_diag

    if debug and isinstance(trace, dict):
        trace["conflicts"] = conflicts_diag
        trace["constraints_final"] = [
            {
                "clause_id": int(getattr(c, "clause_id", 0) or 0),
                "axis": _enum_value_or_none(getattr(c, "axis", None)),
                "direction": _enum_value_or_none(getattr(c, "direction", None)),
                "strength": _enum_value_or_none(getattr(c, "strength", None)),
                "confidence": float(getattr(c, "confidence", 0.0) or 0.0),
                "evidence": getattr(c, "evidence", None),
                "meta": {
                    **(getattr(c, "meta", None) or {}),
                    "axis_family_effective": (getattr(c, "meta", None) or {}).get(
                        "axis_family_effective"
                    ),
                },
            }
            for c in constraints_sorted
        ]

    return NLPResult(
        text=text,
        clauses=tuple(clauses_out) if clauses_out else tuple(clauses_in),
        mentions=mentions_sorted,
        constraints=constraints_sorted,
        diagnostics=diagnostics,
        trace=trace,
    )


def interpret_preference_text(
    text: str,
    *,
    spacy_model: str = "en_core_web_sm",
    clause_config: Optional[ClauseSplitConfig] = None,
    include_xkcd: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """Does: parse free text into structured preference signals.
    Runs NLP parsing, mention extraction, polarity, and constraint detection.
    Returns: intermediate NLP interpretation object.
    """
    res = interpret_nlp(
        text,
        spacy_model=spacy_model,
        clause_config=clause_config,
        include_xkcd=include_xkcd,
        debug=debug,
    )
    return {
        "text": res.text,
        "clauses": [_to_jsonable(c) for c in res.clauses],
        "mentions": [_to_jsonable(m) for m in res.mentions],
        "constraints": [_to_jsonable(c) for c in res.constraints],
        "diagnostics": _to_jsonable(res.diagnostics),
        "trace": _to_jsonable(res.trace),
    }


def build_preference_from_text(
    text: str,
    *,
    spacy_model: str = "en_core_web_sm",
    clause_config: Optional[ClauseSplitConfig] = None,
    include_xkcd: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """Does: end-to-end helper to build preferences directly from text.
    Combines interpretation and normalization steps.
    """
    nlp_res = interpret_nlp(
        text,
        spacy_model=spacy_model,
        clause_config=clause_config,
        include_xkcd=include_xkcd,
        debug=debug,
    )
    return build_preference_from_nlp(nlp_res)
