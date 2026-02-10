<p align="center">
  <img src="assets/demo.png" width="900">
</p>

# Shopping Assistant — Color-Aware Recommendation from Natural Language

This project is a color-aware recommendation system that focuses on the interpretation
and ranking stages of cosmetic shade selection, starting from free-text user preferences.

It parses natural language constraints (e.g. brightness, warmth, exclusions), resolves
color anchors in Lab space, builds adaptive candidate pools, and ranks products using
fully deterministic, test-backed logic. The project is designed as a reproducible
portfolio system, not a production or end-to-end service.

---

## Key features

- Natural language preference interpretation with explicit constraint contracts
- Domain-aware color anchoring and adaptive candidate pooling
- Calibrated Lab-space scoring with deterministic tie-breaking
- Fully test-covered core logic (pytest)
- Streamlit demo application for interactive exploration

---

## Project structure

```
src/Shopping_assistant/     # Core Python package (NLP, color, scoring, recommendation)
Scripts/                   # Offline asset generation and diagnostics (not required at runtime)
data/                      # Versioned runtime assets (lexicons, anchors, calibration)
tests/                     # Pytest-only unit and contract tests
streamlit_app/             # Demo application
```

---

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .
pytest -q
streamlit run streamlit_app/Home.py
```

Optional demo:
```bash
streamlit run streamlit_app/Home.py
```

---

## Data & assets

All runtime assets are versioned under `data/`.
Scripts used to generate or validate these assets are documented in `Scripts/README.md`.

---

## Scope and intent

This repository prioritizes clarity, determinism, and testability over production concerns
(scaling, latency, serving infrastructure).
It is intended to demonstrate applied data science, NLP interpretation, and color-aware
recommendation design.

---

## License

MIT
