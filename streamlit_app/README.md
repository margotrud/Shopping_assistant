# Streamlit App — Shopping Assistant

This directory contains the Streamlit front-end used to explore and demonstrate
the Shopping Assistant color recommendation system.

## Run locally

From the repository root:

```bash
pip install -e .
streamlit run streamlit_app/Home.py
```

## App structure

- `Home.py`  
  Main entry point (landing page).

- `pages/`  
  Secondary pages: playground, explainability, color diagnostics, shade lab, etc.

- `assets/` (if present)  
  Static assets such as images or CSS used by the app.

## Data assets

The app relies on lightweight runtime assets tracked in `data/` (JSON only),
such as:
- color lexicon
- domain color anchors
- scoring and calibration configs

Large datasets, caches, and generated artifacts are intentionally **not**
versioned.

## Notes

- The package must be installed in editable mode (`pip install -e .`)
  for imports to work correctly.
- Some NLP or modeling components may be optional; related tests are skipped
  automatically if dependencies are missing.

## Purpose

This Streamlit app is part of a **portfolio project** showcasing:
- NLP interpretation of user preferences
- color science (Lab space, ΔE, anchors)
- constrained recommendation logic
- end-to-end reproducible experimentation
