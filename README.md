# Shopping Assistant — Text-to-Shade Recommendation

A portfolio project showcasing an end-to-end **NLP → color science → recommendation** pipeline.
The system parses free-text preferences (e.g. *"soft nude pink"*), resolves color anchors,
applies constraint-aware scoring, and returns ranked cosmetic shades.

This repository is intentionally **lightweight and reproducible**: only code and small runtime
JSON assets are versioned.

---

## Installation

```bash
git clone <REPO_URL>
cd <REPO_ROOT>
pip install -e .
```
### Optional dependencies

```bash
pip install -e .[dev]
pip install -e .[embeddings]
pip install -e .[scrape]
```
> Python ≥ 3.10 recommended.

---

## Quick usage (API)

```python
from Shopping_assistant.reco.recommend import recommend_from_text

df = recommend_from_text("I want a soft nude pink lipstick")
print(df.head())
```

---

## Run Streamlit app

```bash
streamlit run streamlit_app/Home.py
```

This launches an interactive demo to explore the recommendation pipeline.

---

## Assets & data policy

Lightweight runtime assets (lexicons, anchors, scoring configs) are versioned under `data/`:
- `data/colors/`
- `data/nlp/`
- `data/models/`

Heavy datasets, caches, generated reports, and experiment artifacts are **intentionally excluded**
from version control.

See `Scripts/README.md` for asset generation and analysis scripts.

---

## Optional: full NLP features

Some NLP components rely on external models.

### spaCy model
```bash
python -m spacy download en_core_web_sm
```

### Sentence Transformers (embeddings)
```bash
pip install -e .[embeddings]
```

If these dependencies are not installed, related tests are automatically skipped and
the core pipeline remains usable.

---

## Tests

```bash
pytest -q
```

Tests requiring heavy models or embeddings are skipped when optional dependencies are missing.

---

## Project status

This project is a **research / portfolio artifact**, not a production system.
The focus is on:
- clean architecture
- reproducible experiments
- explicit constraints and diagnostics
