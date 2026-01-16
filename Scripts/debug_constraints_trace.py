# Scripts/debug_constraints_trace.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Shopping_assistant.nlp.axes.predictor import predict_axis  # noqa: E402
from Shopping_assistant.nlp.parsing import constraints as C  # noqa: E402


def _load_spacy_en():
    import spacy  # noqa: WPS433

    for name in ("en_core_web_md", "en_core_web_lg", "en_core_web_sm"):
        try:
            nlp = spacy.load(name)
            print(f"[spacy] loaded: {name}  lang={nlp.lang}  version={spacy.__version__}")
            return nlp
        except Exception:
            continue
    raise RuntimeError("Install a spaCy English model: python -m spacy download en_core_web_sm")


def _tok_row(tok) -> Dict[str, Any]:
    head = tok.head
    return {
        "i": tok.i,
        "text": tok.text,
        "lemma": tok.lemma_.lower(),
        "pos": tok.pos_,
        "dep": tok.dep_,
        "head": f"{head.text}/{head.dep_}/{head.pos_}" if head is not None else None,
        "is_stop": bool(tok.is_stop),
    }


def _pretty_axis(ax) -> str:
    if ax is None:
        return "None"
    try:
        return ax.value
    except Exception:
        return str(ax)


def _print_predict(tok) -> None:
    queries = C._iter_axis_queries(tok)
    for q in queries:
        pred = predict_axis(
            q,
            context="",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            min_sim=0.35,
            min_margin=0.08,
            debug=True,
        )
        meta = getattr(pred, "meta", {}) or {}
        ranked = meta.get("ranked") or meta.get("ranked_top5") or meta.get("ranked_top3") or []
        best_axis = meta.get("best_axis")

        print(f"    query={q!r}")
        print(
            f"      pred.axis={_pretty_axis(getattr(pred,'axis',None))} "
            f"conf={getattr(pred,'confidence',0.0):.3f} margin={getattr(pred,'margin',0.0):.3f}"
        )
        if best_axis:
            print(f"      meta.best_axis: {best_axis}")
        if ranked:
            print(f"      meta.ranked(top3): {ranked[:3]}")


def _index_kept_constraints(kept) -> Dict[int, List[Any]]:
    """
    Index kept constraints by evidence span so we can attach them back to tokens.
    """
    out: Dict[int, List[Any]] = {}
    for c in kept:
        meta = c.meta or {}
        start = meta.get("evidence_char_start")
        end = meta.get("evidence_char_end")
        if isinstance(start, int) and isinstance(end, int):
            out.setdefault((start, end), []).append(c)
    return out


def trace(text: str) -> None:
    nlp = _load_spacy_en()
    doc = nlp(text)

    kept = C.extract_constraints_from_doc(doc, clause_id=0)
    kept_by_span = _index_kept_constraints(kept)

    print("\n" + "=" * 98)
    print(text)
    print("=" * 98)

    print("\n[PROD KEPT CONSTRAINTS]")
    if not kept:
        print("  (none)")
    else:
        for c in kept:
            m = c.meta or {}
            print(
                f"  - axis={c.axis.value:<10} dir={c.direction.value:<5} strength={c.strength.value:<6} "
                f"conf={c.confidence:.3f} evidence={c.evidence!r} tok={m.get('tok')!r} "
                f"gate={m.get('axis_gate')!r} fam={m.get('axis_family_effective')!r} "
                f"head_is_domain_noun={m.get('head_is_domain_noun')!r}"
            )

    print("\n[TOKENS]")
    for tok in doc:
        row = _tok_row(tok)
        cand = C._is_constraint_candidate(tok)
        print(
            f"[{row['i']:02d}] {row['text']:<12} lemma={row['lemma']:<10} pos={row['pos']:<5} dep={row['dep']:<8} "
            f"head={str(row['head']):<22}  candidate={cand}"
        )

        if not cand:
            continue

        # model-side debug (predictor)
        _print_predict(tok)

        # prod verdict for THIS token: check if any kept constraint evidence span covers token
        tok_start = tok.idx
        tok_end = tok.idx + len(tok.text)
        attached = []
        for (s, e), cs in kept_by_span.items():
            if s <= tok_start and tok_end <= e:
                attached.extend(cs)

        print(f"    prod_verdict: {'KEEP' if attached else 'DROP'}")
        for c in attached:
            m = c.meta or {}
            print(
                f"      -> axis={c.axis.value} dir={c.direction.value} strength={c.strength.value} "
                f"conf={c.confidence:.3f} evidence={c.evidence!r} gate={m.get('axis_gate')!r}"
            )


def main() -> None:
    tests = [
        "I want a berry lipstick but nothing too strong.",
        "I want a brown lipstick, but not dark.",
        "I don't want anything bright.",
        "I want a coral lipstick, not too bright.",
        "I'm looking for something peachy but not loud.",
    ]
    for t in tests:
        trace(t)


if __name__ == "__main__":
    main()
