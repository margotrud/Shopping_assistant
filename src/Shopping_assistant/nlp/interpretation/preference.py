# src/Shopping_assistant/nlp/preference.py
from __future__ import annotations

import re
from dataclasses import is_dataclass, replace
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set

from Shopping_assistant.nlp.axes.predictor import predict_axis
from Shopping_assistant.nlp.llm.analyze_clauses import build_world_alias_index, extract_mentions_free
from Shopping_assistant.nlp.parsing.clauses import ClauseSplitConfig, split_clauses
from Shopping_assistant.nlp.parsing.constraints import extract_constraints_from_clause_text
from Shopping_assistant.nlp.parsing.polarity import (
    decide_clause_polarity,
    infer_polarity_for_mentions,
    make_free_polarity_fn,
)
from Shopping_assistant.nlp.resolve import resolve_preference
from Shopping_assistant.nlp.resolve.conflicts import resolve_symbolic_conflicts
from Shopping_assistant.nlp.resolve.constraint_normalizer import normalize_constraints
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


def _mention_sort_key(m: Mention) -> tuple:
    return (
        int(m.clause_id),
        int(m.span.start),
        str(m.kind.value),
        str(m.canonical),
        str(m.raw),
    )


def _constraint_sort_key(c: Constraint) -> tuple:
    meta = c.meta or {}
    gs = meta.get("evidence_global_start")
    ge = meta.get("evidence_global_end")
    gs_i = int(gs) if isinstance(gs, int) else 10**9
    ge_i = int(ge) if isinstance(ge, int) else 10**9
    return (
        int(c.clause_id),
        str(c.axis.value),
        str(c.direction.value),
        str(c.strength.value),
        gs_i,
        ge_i,
        str(c.evidence),
    )


def _normalize_model_name(model_name: str) -> str:
    if "/" in (model_name or ""):
        return model_name
    return f"sentence-transformers/{model_name}"


@lru_cache(maxsize=8192)
def _is_axis_like_token(
    token: str,
    *,
    model_name: str,
    min_sim: float,
    min_margin: float,
) -> bool:
    t = (token or "").strip().lower()
    if not t or len(t) <= 2:
        return False

    pred = predict_axis(
        t,
        model_name=_normalize_model_name(model_name),
        min_sim=float(min_sim),
        min_margin=float(min_margin),
        debug=False,
    )
    return pred.axis is not None


def _blocked_lemmas_from_mentions(
    mention_names: List[str],
    *,
    axis_model: str,
    axis_min_sim: float,
    axis_min_margin: float,
) -> Set[str]:
    blocked: Set[str] = set()
    for name in mention_names:
        for tok in _TOKEN_SPLIT_RE.split((name or "").lower()):
            if not tok:
                continue
            if _is_axis_like_token(
                tok,
                model_name=axis_model,
                min_sim=axis_min_sim,
                min_margin=axis_min_margin,
            ):
                continue
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


def interpret_nlp(
    text: str,
    *,
    spacy_model: str = "en_core_web_sm",
    clause_config: Optional[ClauseSplitConfig] = None,
    include_xkcd: bool = True,
    debug: bool = False,
) -> NLPResult:
    nlp = load_spacy(spacy_model)

    cfg = clause_config or _DEFAULT_CLAUSE_CFG
    clauses_in = split_clauses(
        text,
        nlp=nlp,
        spacy_model=spacy_model,
        config=cfg,
        debug=debug,
    )

    color_index = build_world_alias_index(include_xkcd=include_xkcd)
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

        blocked_lemmas = _blocked_lemmas_from_mentions(
            mention_names,
            axis_model="all-MiniLM-L6-v2",
            axis_min_sim=0.35,
            axis_min_margin=0.08,
        )

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
                meta={"hue_deg": m.get("hue_deg"), "chunk": cl.meta.get("chunk")},
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
                        "span": {"start": int(mention_obj.span.start), "end": int(mention_obj.span.end)},
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
                        "clause_id": c2.clause_id,
                        "axis": c2.axis.value,
                        "direction": c2.direction.value,
                        "strength": c2.strength.value,
                        "confidence": float(getattr(c2, "confidence", 0.0) or 0.0),
                        "evidence": c2.evidence,
                        "meta": c2.meta,
                    }
                )

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
    constraints_final = normalize_constraints(constraints_final, strict=debug)

    constraints_sorted = tuple(sorted(constraints_final, key=_constraint_sort_key))
    mentions_sorted = tuple(sorted(all_mentions, key=_mention_sort_key))

    diagnostics = dict(diagnostics or {})
    diagnostics["conflicts"] = conflicts_diag

    if debug and isinstance(diagnostics.get("mentions"), list):
        diagnostics["mentions"] = sorted(
            diagnostics["mentions"],
            key=lambda d: (
                d.get("clause_id", 0),
                (d.get("span", {}) or {}).get("start", 10**9),
                d.get("canonical", ""),
                d.get("raw", ""),
            ),
        )
    if debug and isinstance(diagnostics.get("constraints"), list):
        diagnostics["constraints"] = sorted(
            diagnostics["constraints"],
            key=lambda d: (
                d.get("clause_id", 0),
                str(d.get("axis", "")),
                str(d.get("direction", "")),
                str(d.get("strength", "")),
                (d.get("meta", {}) or {}).get("evidence_global_start", 10**9),
            ),
        )

    if debug and isinstance(trace, dict):
        trace["conflicts"] = conflicts_diag
        trace["constraints_final"] = [
            {
                "clause_id": c.clause_id,
                "axis": c.axis.value,
                "direction": c.direction.value,
                "strength": c.strength.value,
                "confidence": float(getattr(c, "confidence", 0.0) or 0.0),
                "evidence": c.evidence,
                "meta": c.meta,
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
