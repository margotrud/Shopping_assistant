# Shopping Assistant V8 — NLP-to-Scoring Lipstick Recommender

## Project purpose

This is a **portfolio project** designed to demonstrate:
- NLP interpretation of free-text preferences
- Projection of semantic intents onto numeric axes
- Constraint-based scoring and ranking
- Deterministic, testable ML-style pipelines

It is **not** intended to be a production system or deployed service.

---

## What it does

- Parses natural language preferences (likes, dislikes, modifiers)
- Resolves polarity and intent at clause level
- Projects intents onto numeric color axes (brightness, depth, vibrancy, clarity, etc.)
- Builds numeric threshold constraints from resolved intents
- Scores and ranks lipstick shades from a real Sephora inventory
- Outputs reproducible top‑K recommendations

---

## High-level pipeline

```
User text
   ↓
Clause splitting & polarity detection
   ↓
Intent resolution (likes / dislikes)
   ↓
Axis projection & merge
   ↓
Numeric thresholds
   ↓
Constraint-based scoring
   ↓
Ranked shades (CSV + console)
```

---

## Repository structure

```
src/Shopping_assistant/
├── nlp/
│   ├── interpretation/     # clause parsing, polarity, mentions
│   ├── resolve/            # axis projection, merge, thresholds, adapters
│   └── schema.py           # shared NLP dataclasses
├── color/
│   ├── scoring.py          # constraint evaluation & scoring
│   └── features.py         # numeric color features
├── reco/
│   └── recommend.py        # end-to-end recommendation entrypoint
├── io/
│   └── assets.py           # inventory & calibration loading
tests/
│   ├── test_*contract.py   # functional contracts
│   ├── test_*invariants.py # determinism & stability checks
data/
├── inventory/              # enriched Sephora lipstick data (CSV)
└── reports/                # demo outputs
scripts/
└── demo_reco.py             # one-command demo
```

---

## Installation

From the repository root:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -U pip
pip install -e .
```

### Optional NLP dependencies (spaCy)

Some NLP components rely on spaCy.

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

If spaCy is not installed, the code fails explicitly via guarded imports.

---

## Run a demo

One command demo:

```bash
python scripts/demo_reco.py --text "I want a deep red lipstick but not too bright"
```

What it does:
- Runs the full NLP → scoring pipeline
- Prints a top‑10 table to the console
- Writes `data/reports/demo_top10.csv`

If no `--text` is provided, the script runs several representative examples.

---

## Run tests

```bash
pytest -q
```

Test suite focuses on:
- Deterministic scoring
- Contract-level behavior
- Invariant preservation across refactors

---

## Key files to review

- `src/Shopping_assistant/reco/recommend.py`  
  End‑to‑end recommendation orchestration.

- `src/Shopping_assistant/nlp/resolve/`  
  Axis projection, intent merge, threshold logic.

- `src/Shopping_assistant/color/scoring.py`  
  Constraint evaluation and ranking mechanics.

- `tests/`  
  Contract tests and scoring invariants.

---

## Notes

- The project prioritizes **clarity, determinism, and testability**
- No UI is provided; focus is on pipeline logic and architecture
- All data and artifacts are included to allow full reproducibility
