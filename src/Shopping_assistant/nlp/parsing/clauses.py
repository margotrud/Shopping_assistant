# src/Shopping_assistant/nlp/clauses.py
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, TypedDict

from Shopping_assistant.nlp.schema import Clause, Polarity
from Shopping_assistant.utils.optional_deps import require

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Span, Token
else:
    Language = Any  # type: ignore
    Doc = Any       # type: ignore
    Span = Any      # type: ignore
    Token = Any     # type: ignore


class ClauseChunk(TypedDict, total=False):
    """
    Represents a clause-like chunk extracted from the input text.
    Fields: text, start_char, end_char, reasons, kind.
    """
    text: str
    start_char: int
    end_char: int
    reasons: List[str]
    kind: str


class ClauseSplitConfig(TypedDict):
    """Data-driven config: UD tags/deps + punctuation only."""
    split_pos: List[str]
    split_dep: List[str]
    split_punct: List[str]
    keep_len_min: int
    keep_len_max: Optional[int]


@lru_cache(maxsize=2)
def _load_spacy(model_name: str) -> Language:
    spacy = require(
        "spacy",
        extra="spacy",
        purpose="Needed for nlp.clauses.split_clauses_with_reasons().",
    )
    try:
        return spacy.load(model_name)
    except OSError as e:
        raise RuntimeError(
            f"spaCy model {model_name!r} not available. "
            f"Install it with: python -m spacy download {model_name}"
        ) from e


def _has_finite_verb(span: Span) -> bool:
    for t in span:
        if t.pos_ in ("VERB", "AUX"):
            vf = t.morph.get("VerbForm")
            if not vf or "Fin" in vf:
                return True
            if not any(v in ("Inf", "Part", "Ger", "Conv") for v in vf):
                return True
    return False


def _has_neg_mark(span: Span) -> bool:
    return any(t.dep_ == "neg" for t in span)


def _has_content_head(span: Span) -> bool:
    return any(t.pos_ in ("ADJ", "NOUN", "PROPN") for t in span)


def _has_overt_subject(span: Span) -> bool:
    return any(t.dep_ in ("nsubj", "nsubjpass", "csubj", "expl") for t in span)


def _is_wh_interrogative(tok: Token) -> bool:
    if tok.pos_ not in ("ADV", "PRON", "SCONJ"):
        return False
    if tok.tag_.startswith("W"):
        return True
    pron_types = tok.morph.get("PronType")
    return any("Int" in pt for pt in pron_types)


def _is_corrective_parenthetical_neg(inside: Span) -> bool:
    neg_tokens = [t for t in inside if t.dep_ == "neg"]
    if not neg_tokens:
        return False

    for t in inside:
        if t.pos_ in ("VERB", "AUX"):
            vf = t.morph.get("VerbForm")
            if vf and "Fin" in vf:
                return False

    if not any(t.pos_ in ("ADJ", "NOUN") for t in inside):
        return False

    content = [t for t in inside if not t.is_punct and not t.is_space]
    if len(content) > 4:
        return False

    if any(t.dep_ == "cc" for t in inside):
        return True

    return bool(content) and content[0].dep_ == "neg"


def _is_elliptical_corrective_right(span: Span) -> bool:
    return (not _has_finite_verb(span)) and _has_neg_mark(span) and _has_content_head(span)


def _sents_merged_on_leading_connective(doc: Doc) -> List[Span]:
    """
    Does:
        Merge consecutive sents when a sent starts with a clause-link token.
    Returns:
        Spans covering merged ranges.
    """
    sents = [s for s in doc.sents if s.text.strip()]
    if not sents:
        return []

    merged: List[Span] = []
    cur_start = sents[0].start
    cur_end = sents[0].end

    for s in sents[1:]:
        t0 = s[0]
        starts_with_link = (t0.pos_ in ("CCONJ", "SCONJ")) or (t0.dep_ in ("cc", "mark"))

        # structural exception: initial discourse CCONJ + WH interrogative => do not merge
        if starts_with_link and t0.pos_ == "CCONJ" and t0.text[:1].isupper() and len(s) > 1:
            j = 1
            while j < len(s) and s[j].is_punct:
                j += 1
            if j < len(s) and _is_wh_interrogative(s[j]):
                starts_with_link = False

        if starts_with_link:
            cur_end = s.end
        else:
            merged.append(doc[cur_start:cur_end])
            cur_start, cur_end = s.start, s.end

    merged.append(doc[cur_start:cur_end])
    return merged


def _is_clause_boundary_token(tok: Token, cfg: ClauseSplitConfig, span: Span) -> Optional[str]:
    doc = tok.doc
    span_start, span_end = span.start, span.end

    def _left_right(_tok: Token) -> Tuple[Span, Span]:
        left = doc[span_start:_tok.i]
        right = doc[_tok.i + 1:span_end]
        return left, right

    if tok.dep_ in ("cc", "mark"):
        if tok.dep_ == "cc" and tok.i + 1 < span_end:
            nxt = doc[tok.i + 1]
            if _is_wh_interrogative(nxt):
                return None

        left, right = _left_right(tok)
        if _has_finite_verb(left) and _has_finite_verb(right):
            if _has_overt_subject(right):
                return f"{tok.dep_.upper()} boundary between finite-verb segments"
            return None

    if tok.pos_ in cfg["split_pos"]:
        if _is_wh_interrogative(tok):
            return None

        left, right = _left_right(tok)
        if _has_finite_verb(left) and _has_finite_verb(right):
            return f"{tok.pos_} boundary between finite-verb segments"

        if tok.pos_ == "CCONJ":
            right2 = doc[tok.i + 1:span_end]
            if _is_elliptical_corrective_right(right2):
                return "CCONJ with NEG elliptical (corrective)"

        return None

    if tok.is_punct and (tok.text in cfg["split_punct"] or tok.text in {"(", ")", "[", "]"}):
        left = doc[span_start:tok.i]
        right = doc[tok.i + 1:span_end]

        if tok.text == "," and any(t.dep_ == "mark" for t in left) and not _has_finite_verb(left):
            return None

        if tok.text in {"(", "["}:
            close_i = None
            for j in range(tok.i + 1, span_end):
                if doc[j].text in {")", "]"}:
                    close_i = j
                    break
            if close_i is None:
                close_i = span_end
            inside = doc[tok.i + 1:close_i]
            if _is_corrective_parenthetical_neg(inside):
                return "PUNCT parenthetical NEG (corrective)"

        if tok.text in {")", "]"}:
            open_i = None
            for j in range(tok.i - 1, span_start - 1, -1):
                if doc[j].text in {"(", "["}:
                    open_i = j
                    break
            if open_i is not None:
                inside = doc[open_i + 1:tok.i]
                if _is_corrective_parenthetical_neg(inside):
                    return "PUNCT end parenthetical NEG (corrective)"

        if tok.dep_ in ("relcl", "appos") or tok.head.dep_ in ("relcl", "appos"):
            return None

        nxt = doc[tok.i + 1] if tok.i + 1 < len(doc) else None
        if nxt is not None and (nxt.pos_ in cfg["split_pos"] or nxt.dep_ in ("cc", "mark")):
            if tok.text not in {"(", "["}:
                return None

        if any(t.dep_ == "relcl" for t in right) or tok.head.dep_ == "relcl":
            return None

        if any(t.dep_ == "mark" for t in left) and _has_finite_verb(left):
            return "PUNCT after fronted subclause (mark)"

        if _has_finite_verb(left) and _has_finite_verb(right):
            first_fin = None
            for j, t in enumerate(right):
                if t.pos_ in ("VERB", "AUX"):
                    vf = t.morph.get("VerbForm")
                    if (not vf) or ("Fin" in vf) or (not any(v in ("Inf", "Part", "Ger", "Conv") for v in vf)):
                        first_fin = j
                        break
            if first_fin is not None:
                if any((t.pos_ in cfg["split_pos"] or t.dep_ in ("cc", "mark")) for t in right[:first_fin]):
                    return None
            return "PUNCT separating two finite-verb segments"

        clausal_deps = set(cfg["split_dep"])
        head_dep = tok.head.dep_
        child_deps = {ch.dep_ for ch in tok.children}
        deps_all = {head_dep} | child_deps
        deps_clausal = deps_all & clausal_deps

        if deps_clausal:
            if deps_clausal == {"conj"}:
                return None
            if _has_finite_verb(right) or (_has_neg_mark(right) and _has_content_head(right)):
                return f"PUNCT with clausal dep {head_dep or 'child'}"
            return None

        if _has_finite_verb(left) and (not _has_finite_verb(right)) and _has_neg_mark(right) and _has_content_head(right):
            return "PUNCT with NEG elliptical (corrective)"

    return None


def _materialize(span: Span, cut_idxs: List[int], reasons_by_cut: Dict[int, List[str]]) -> List[ClauseChunk]:
    """
    Does:
        Turn token cuts into text slices and trim clause-link tokens.
    Returns:
        List[ClauseChunk] with offsets + reasons.
    """
    if not cut_idxs:
        a, b = span.start, span.end
        while a < b:
            t0 = span.doc[a]
            # drop leading coordinating linkers only (CCONJ/cc); keep SCONJ/mark (fronted subclauses)
            if t0.pos_ == "CCONJ" or t0.dep_ == "cc":
                a += 1
                continue
            break

        if a >= b:
            return []
        sub = span.doc[a:b]
        txt = sub.text.strip(" ,;:.—–()[]")
        if not txt:
            return []
        return [{
            "text": txt,
            "start_char": sub.start_char,
            "end_char": sub.end_char,
            "reasons": [],
        }]

    idxs = [span.start] + sorted(set(i for i in cut_idxs if span.start < i < span.end)) + [span.end]
    out: List[ClauseChunk] = []

    for a, b in zip(idxs, idxs[1:]):
        while a < b:
            t_last = span.doc[b - 1]
            if _is_wh_interrogative(t_last):
                break
            if t_last.pos_ in ("CCONJ", "SCONJ") or t_last.dep_ in ("cc", "mark"):
                b -= 1
                continue
            break
        if a >= b:
            continue

        is_first_chunk = (a == span.start)
        if not is_first_chunk:
            while a < b:
                t0 = span.doc[a]
                if t0.is_punct and t0.text in {",", ";", ":", "-", "—", "–"}:
                    a += 1
                    continue
                break
            if a >= b:
                continue

            while a < b:
                t0 = span.doc[a]
                if t0.pos_ == "CCONJ" and t0.text[:1].isupper():
                    break
                if _is_wh_interrogative(t0):
                    break
                if t0.pos_ in ("CCONJ", "SCONJ") or t0.dep_ in ("cc", "mark"):
                    a += 1
                    continue
                break
            if a >= b:
                continue

        sub = span.doc[a:b]
        text_trimmed = sub.text.strip(" ,;:.—–()[]")
        if not text_trimmed:
            continue

        lead_ws = len(sub.text) - len(sub.text.lstrip())
        end_char = sub.start_char + len(sub.text.rstrip())

        ch: ClauseChunk = {
            "text": text_trimmed,
            "start_char": sub.start_char + lead_ws,
            "end_char": end_char,
            "reasons": [],
        }

        if not is_first_chunk:
            rs = reasons_by_cut.get(a, [])
            if rs:
                ch["reasons"] = list(rs)

        out.append(ch)

    return out


def _normalize_parenthetical_not(chunks: List[ClauseChunk]) -> None:
    for ch in chunks:
        txt = ch["text"]
        if " (not " in txt:
            ch["text"] = txt.replace(" (not ", ", not ")


def _tag_elliptical_neg(chunks: List[ClauseChunk]) -> None:
    for ch in chunks:
        rs = " ".join(ch.get("reasons") or [])
        if "NEG elliptical" in rs or "parenthetical NEG" in rs:
            ch["kind"] = "ELLIPTICAL_NEG"


def split_clauses_with_reasons(
    text: str,
    *,
    nlp: Language | None = None,
    spacy_model: str = "en_core_web_sm",
    config: ClauseSplitConfig,
    debug: bool = False,
) -> List[ClauseChunk]:
    """
    Does:
        Split English text into clause-like chunks using UD POS/DEP & punctuation.
    Returns:
        List[ClauseChunk] with offsets and cut reasons.

    Important:
        If `nlp` is provided, it is used directly and spaCy is NOT loaded again.
    """
    if debug:
        log.debug("ENTER split_clauses_with_reasons text=%r", text)

    if nlp is None:
        nlp = _load_spacy(spacy_model)
    doc = nlp(text)

    spans = _sents_merged_on_leading_connective(doc)
    if not spans:
        return []

    all_chunks: List[ClauseChunk] = []

    for sent_span in spans:
        cut_idxs: List[int] = []
        reasons_by_cut: Dict[int, List[str]] = {}

        for tok in sent_span:
            reason = _is_clause_boundary_token(tok, config, sent_span)
            if reason is None:
                continue

            cut_i = tok.i + 1
            if not (sent_span.start < cut_i < sent_span.end):
                continue

            cut_idxs.append(cut_i)
            reasons_by_cut.setdefault(cut_i, []).append(reason)

        chunks = _materialize(sent_span, cut_idxs, reasons_by_cut)
        all_chunks.extend(chunks)

    _normalize_parenthetical_not(all_chunks)
    _tag_elliptical_neg(all_chunks)

    out: List[ClauseChunk] = []
    for ch in all_chunks:
        if len(ch["text"]) < config["keep_len_min"]:
            continue
        mx = config["keep_len_max"]
        if mx is not None and len(ch["text"]) > mx:
            continue
        out.append(ch)

    return out


def chunks_to_clauses(chunks: List[ClauseChunk]) -> Tuple[Clause, ...]:
    out: List[Clause] = []
    for i, ch in enumerate(chunks):
        text = ch.get("text", "")
        kind = ch.get("kind")
        reasons = ch.get("reasons") or []
        reason = reasons[0] if reasons else None
        elliptical_neg = (kind == "ELLIPTICAL_NEG")

        out.append(
            Clause(
                clause_id=i,
                text=text,
                polarity=Polarity.UNKNOWN,
                elliptical_neg=elliptical_neg,
                reason=reason,
                meta={"chunk": ch},
            )
        )

    return tuple(out)


def split_clauses(
    text: str,
    *,
    nlp: Language | None = None,
    spacy_model: str = "en_core_web_sm",
    config: ClauseSplitConfig,
    debug: bool = False,
) -> Tuple[Clause, ...]:
    chunks = split_clauses_with_reasons(
        text,
        nlp=nlp,
        spacy_model=spacy_model,
        config=config,
        debug=debug,
    )
    return chunks_to_clauses(chunks)


__all__ = [
    "ClauseChunk",
    "ClauseSplitConfig",
    "split_clauses_with_reasons",
    "chunks_to_clauses",
    "split_clauses",
]
