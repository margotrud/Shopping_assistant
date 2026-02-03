# Shopping Assistant V8 (Lipstick Recommender)

Portfolio project combining **NLP**, **color science**, and **recommendation systems**
to suggest cosmetic shades (lipsticks) based on natural language preferences.

This repository is intentionally kept **lightweight** and **reviewable**:
large datasets, trained models, and generated artifacts are excluded.

---

## Installation

```bash
python -m venv .venv
# activate the virtual environment
pip install -e .
```

The project uses a `src/` layout and must be installed (editable mode recommended).

---

## Optional dependency: sentence-transformers

Some NLP polarity and constraint features rely on embedding-based similarity
(`sentence-transformers`).

Install it only if you want full NLP behavior and to run all tests:

```bash
pip install sentence-transformers
```

If not installed:
- Core recommendation still works
- Related NLP tests are automatically skipped

---

## Run tests

```bash
pytest -q
```

---

## Example usage

```bash
python -c """
from Shopping_assistant.reco.recommend import recommend_from_text

df = recommend_from_text(
    'I want a red lipstick but not too bright',
    topk=5,
)

print(df.head())
"""
```

---

## Data policy

This repository **does not include**:
- Full product inventories (CSV)
- Training datasets
- Trained ML models
- Generated plots or reports

Only **runtime configuration assets** are versioned:
- Color and NLP lexicons (`data/colors/*.json`, `data/nlp/*.json`)
- Scoring and calibration configs (`data/models/*.json`)

This keeps the repo:
- Fast to clone
- Easy to review
- Suitable for GitHub and portfolio use

---

## Scope

This is a **portfolio / research project**, not a production system.
Design decisions favor:
- Transparency
- Explicit heuristics
- Testable behavior
- Reproducible logic
