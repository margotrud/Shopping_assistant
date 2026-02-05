# Scripts/debug_constraints_forensic.py
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Shopping_assistant.nlp.parsing import constraints as C  # noqa: E402
from Shopping_assistant.nlp.schema import Polarity  # noqa: E402


def _load_spacy_en():
    import spacy  # noqa: WPS433

    for name in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
        try:
            nlp = spacy.load(name)
            meta = getattr(nlp, "meta", {}) or {}
            print(f"[spacy] loaded: {name}  lang={meta.get('lang')}  version={meta.get('version')}")
            return nlp
        except Exception:
            continue
    raise RuntimeError("Install a spaCy English model: python -m spacy download en_core_web_sm")


class AxisCallLog:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def wrapped(self, text: str, *, context: str, model_name: str, min_sim: float, min_margin: float, debug: bool):
        # Call the real function (keep a handle)
        pred = self._real(
            text,
            context=context,
            model_name=model_name,
            min_sim=min_sim,
            min_margin=min_margin,
            debug=debug,
        )
        meta = getattr(pred, "meta", None) or {}
        ranked = meta.get("ranked")
        best_axis = meta.get("best_axis")

        self.calls.append(
            {
                "text": text,
                "context": context,
                "model_name": model_name,
                "min_sim": float(min_sim),
                "min_margin": float(min_margin),
                "debug": bool(debug),
                "pred_axis": pred.axis.value if pred.axis else None,
                "conf": float(pred.confidence),
                "margin": float(pred.margin),
                "best_axis": best_axis,
                "ranked_top5": ranked[:5] if ranked else None,
            }
        )
        return pred

    def install(self):
        # constraints.py does: from ...predictor import predict_axis
        # so we patch the symbol in module C
        self._real = C.predict_axis
        C.predict_axis = self.wrapped  # type: ignore[assignment]

    def uninstall(self):
        C.predict_axis = self._real  # type: ignore[assignment]


def run_one(text: str, clause_id: int) -> None:
    nlp = _load_spacy_en()

    print("\n" + "=" * 110)
    print(text)
    print("=" * 110)
    print(f"[import] constraints path: {getattr(C, '__file__', '<?>')}")

    log = AxisCallLog()
    log.install()
    try:
        # IMPORTANT: run the real pipeline function only (no extra predict_axis calls)
        out = C.extract_constraints_from_clause_text(
            text,
            clause_id=clause_id,
            clause_polarity=Polarity.UNKNOWN,
            blocked_lemmas=None,
            nlp=nlp,
            mapper_model="all-MiniLM-L6-v2",
            mapper_threshold=0.35,
            mapper_min_margin=0.08,
        )
    finally:
        log.uninstall()

    print("\n[FORWARD OUTPUT]")
    if not out:
        print("  (no constraints)")
    else:
        for c in out:
            print(
                f"  - axis={c.axis.value:<10} dir={c.direction.value:<5} strength={c.strength.value:<6} "
                f"conf={c.confidence:.3f} evidence={c.evidence!r}"
            )
            m = c.meta or {}
            keep = {k: m.get(k) for k in ("axis_source", "axis_query", "axis_score", "axis_margin", "tok", "negated")}
            print(f"    meta={keep}")

    print("\n[ACTUAL predict_axis CALLS MADE BY constraints.py]")
    if not log.calls:
        print("  (no predict_axis calls)  -> means constraints dropped all candidates BEFORE axis mapping.")
        return

    for i, call in enumerate(log.calls, start=1):
        # Show the exact text passed + decision vs thresholds
        conf = call["conf"]
        margin = call["margin"]
        ms = call["min_sim"]
        mm = call["min_margin"]
        gate = (conf >= ms) and (margin >= mm)
        print(
            f"  [{i:02d}] text={call['text']!r}  axis={call['pred_axis']}  conf={conf:.3f}  margin={margin:.3f}  "
            f"gate={'PASS' if gate else 'FAIL'} (min_sim={ms}, min_margin={mm})"
        )
        if call["best_axis"] or call["ranked_top5"]:
            print(f"       best_axis={call['best_axis']} ranked_top5={call['ranked_top5']}")


def main() -> None:
    tests = [
        "I want a very soft nude lipstick.",
        "I want a coral lipstick, not too bright.",
        "I want a berry lipstick but nothing too strong.",
        "I want a brown lipstick, but not dark.",
        "I don't want anything bright.",
        "Not neon. Definitely not neon.",
        "I'm looking for something peachy but not loud.",
    ]
    for i, t in enumerate(tests, start=1):
        run_one(t, clause_id=i)


if __name__ == "__main__":
    main()
