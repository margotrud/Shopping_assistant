# Shopping Assistant — NLP-Driven Preference Resolution & Product Ranking

## Overview

This project demonstrates an **end-to-end data science pipeline** that transforms **free-form natural language preferences** into **numeric constraints** and applies them to **rank real products**.

The focus is **not UI or deployment**, but:
- robust NLP interpretation,
- explicit preference resolution,
- interpretable numeric modeling,
- deterministic scoring,
- and testable engineering.

The domain used for demonstration is **lipstick recommendation**, chosen for its **high semantic ambiguity** (color, intensity, exclusions, trade-offs).

---

## What This Project Demonstrates (for Recruiters)

**Core DS / ML Engineering skills:**
- Natural language → structured representation
- Preference resolution under ambiguity and negation
- Mapping language to numeric axes
- Multi-constraint scoring and ranking
- Deterministic, testable pipelines
- Data enrichment and calibration
- Clear separation between modeling, data, and orchestration

**What it deliberately does NOT try to be:**
- A production product
- A deployed system
- A UI-heavy demo

This is a **technical portfolio project**, optimized for **code review**, **reasoning clarity**, and **signal density**.

---

## High-Level Pipeline

```
User text
  ↓
NLP interpretation
  ↓
Preference resolution (likes / dislikes / intensity)
  ↓
Projection to numeric axes (brightness, depth, saturation, etc.)
  ↓
Constraint merging & thresholds
  ↓
Deterministic scoring against enriched product inventory
  ↓
Ranked recommendations
```

---

## Example

**Input**
```
"I want a deep red lipstick but not too bright"
```

**Output**
- Structured constraints on color axes (depth ↑, brightness ↓)
- Ranked list of matching shades
- Deterministic ordering (reproducible across runs)

---

## Repository Structure

```
src/Shopping_assistant/
│
├── nlp/                  # NLP interpretation & preference resolution
│   ├── interpretation/   # Clause parsing, mention extraction
│   ├── resolve/          # Axis projection, merge logic, thresholds
│   └── schema.py         # Typed NLP data structures
│
├── color/
│   ├── scoring.py        # Numeric scoring engine
│   ├── distance.py       # Color distance functions
│   └── calibration/      # Calibration utilities
│
├── reco/                 # Recommendation orchestration
│   └── recommend.py
│
├── ml/                   # Calibration, clustering, analysis scripts
├── io/                   # Asset loading & validation
├── utils/                # Shared utilities
│
Scripts/
├── demo_reco.py          # One-command end-to-end demo
├── reco_ab.py            # A/B & diagnostic experiments
│
data/
├── enriched_data/        # Product inventory with numeric axes
├── calibration/          # Axis calibration files
├── prototypes/           # Color prototypes
└── assignments/          # Prototype ↔ product mappings
│
tests/
├── test_scoring_invariants.py
├── test_color_ranking_goldens.py
├── test_reco_contract.py
└── ...
```

---

## Data

The project ships with **real, pre-enriched data** to ensure full reproducibility.

- **Enriched inventory**  
  Lipstick shades with numeric representations:
  - color spaces (Lab / HSL / HSV)
  - derived axes (brightness, depth, vibrancy, clarity, etc.)

- **Calibration files**  
  JSON files defining:
  - axis cutpoints
  - strength mappings
  - scoring weights

- **Prototypes & assignments**  
  Used for clustering and interpretability experiments.

No external APIs or live scraping are required to run the pipeline.

---

## Quickstart (Reproducible)

### 1. Install
```bash
pip install -e .
```

### 2. Run demo
```bash
python Scripts/demo_reco.py   --text "I want a deep red lipstick but not too bright"
```

### 3. Run tests
```bash
pytest -q
```

---

## Design Principles

- **Deterministic by default**  
  Same input → same output.

- **Explicit over implicit**  
  No hidden heuristics, no black-box magic.

- **Numeric first**  
  Language is translated into numeric constraints as early as possible.

- **Test-driven**  
  Ranking stability and invariants are enforced by tests.

- **Debuggable**  
  Intermediate representations are inspectable at every step.

---

## Why Lipstick?

Lipstick is a **hard NLP problem**, not a toy domain:
- dense adjectives (“deep”, “bright”, “soft”, “muted”)
- frequent negations (“not too bright”)
- subjective trade-offs
- overlapping color semantics

This makes it ideal to demonstrate **preference modeling**, **constraint resolution**, and **ranking under ambiguity**.

---

## Limitations

- English-only NLP
- Offline, pre-enriched dataset
- No user personalization or learning loop
- No production deployment layer

---

## Status

- ✔ End-to-end pipeline functional
- ✔ Fully reproducible
- ✔ Covered by tests
- ✔ Suitable for technical review

---

## Author

**Margot**  
Data Scientist / ML Engineer  

This project was built as a **technical portfolio** to demonstrate applied NLP, numeric modeling, and recommendation logic.
