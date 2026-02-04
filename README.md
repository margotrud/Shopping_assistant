# Shopping Assistant V8 (Lipstick Recommender)

Portfolio project combining **NLP**, **color science**, and **recommendation systems**
to suggest cosmetic shades (lipsticks) based on natural language preferences.

This repository is intentionally kept **lightweight** and **reviewable**:
large datasets, trained models, and generated artifacts are excluded.

---

## Installation

```bash
git clone <repo-url>
cd pythonProject
pip install -e .
python -m spacy download en_core_web_sm
```

### Optional dependencies

```bash
pip install -e .[dev]
pip install -e .[embeddings]
pip install -e .[scrape]
```

The project uses a `src/` layout and must be installed
(editable mode recommended for development).

---

## Optional dependency: sentence-transformers

Some NLP polarity and constraint features rely on embedding-based similarity
(`sentence-transformers`).

Install it **only** if you want full NLP behavior and to run all tests:

```bash
pip install sentence-transformers
```

If not installed:
- Core recommendation logic still works
- Embedding-dependent NLP tests are automatically skipped

---

## Run tests

```bash
pytest -q
```

---

## Example usage

```bash
python -c "
from Shopping_assistant.reco.recommend import recommend_from_text

df = recommend_from_text(
    'I want a red lipstick but not too bright',
    topk=5,
)

print(df.head())
"
```

---

## Data policy

This repository **does not include**:
- Full product inventories (CSV / scraped data)
- Training datasets
- Trained ML models
- Generated plots or reports
- Cached product images

Only **runtime configuration assets** are versioned:
- Color and NLP lexicons (`data/colors/*.json`, `data/nlp/*.json`)
- Scoring and calibration configs (`data/models/*.json`)

This keeps the repository:
- Fast to clone
- Easy to review
- Suitable for GitHub and portfolio evaluation

---

## Scope

This is a **portfolio / research project**, not a production system.

Design decisions favor:
- Transparency
- Explicit heuristics
- Testable behavior
- Reproducible logic
