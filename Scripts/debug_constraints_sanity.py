# Scripts/debug_constraints_sanity.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# -----------------------------
# Bootstrap: make src importable
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]  # project root
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Shopping_assistant.nlp.parsing.constraints import extract_constraints_from_clause_text  # noqa: E402
from Shopping_assistant.nlp.schema import Polarity  # noqa: E402


def _load_spacy_en():
    import spacy  # noqa: WPS433

    for name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
        try:
            return spacy.load(name)
        except Exception:
            continue
    raise RuntimeError(
        "No spaCy English model found. Install one, e.g.\n"
        "  python -m spacy download en_core_web_sm\n"
    )


def _fmt_constraints(cs) -> str:
    if not cs:
        return "  (no constraints)"
    lines: List[str] = []
    for c in cs:
        lines.append(
            "  - "
            f"axis={c.axis.value:<11} "
            f"dir={c.direction.value:<5} "
            f"strength={c.strength.value:<6} "
            f"conf={c.confidence:0.3f} "
            f"evidence='{c.evidence}'"
        )
    return "\n".join(lines)


def main() -> None:
    nlp = _load_spacy_en()

    tests = [
        "I want a very soft nude lipstick.",
        "I want a coral lipstick, not too bright.",
        "I want a berry lipstick but nothing too strong.",
        "I want a brown lipstick, but not dark.",
        "I don't want anything bright.",
        "Not neon. Definitely not neon.",
        "I'm looking for something peachy but not loud.",
    ]

    print("\n=== Constraints sanity check ===\n")
    for i, text in enumerate(tests, 1):
        cs = extract_constraints_from_clause_text(
            text,
            clause_id=i,
            clause_polarity=Polarity.UNKNOWN,
            blocked_lemmas=None,
            nlp=nlp,
            mapper_model="all-MiniLM-L6-v2",
            mapper_threshold=0.35,
            mapper_min_margin=0.08,
        )
        print(f"[{i:02d}] {text}")
        print(_fmt_constraints(cs))
        print()

    print("Done.")


if __name__ == "__main__":
    main()
