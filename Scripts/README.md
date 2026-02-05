# Scripts

This directory contains offline scripts used to build, validate, and diagnose the runtime assets
consumed by the Shopping_assistant package. These scripts are not required at inference time.

## Asset generation

| Asset / Output file | Script | Description |
|---------------------|--------|-------------|
| `data/nlp/color_lexicon.json` | `build/build_color_lexicon.py` | Builds the normalized color lexicon used by NLP parsing and interpretation. |
| `data/nlp/domain_color_anchors.json` | `build/build_domain_color_anchors.py` | Computes domain-specific color anchors from inventory statistics. |
| `data/models/color_scoring_calibration.json` | `build/build_color_scoring_calibration.py` | Fits and exports scoring calibration parameters. |
| `data/models/color_preference_weights.json` | `train/train_preference_weights.py` | Trains and exports preference weighting coefficients. |

## Diagnostics / QA

Scripts under `diagnostics/` are used to validate coverage, thresholds, and failure modes
of the recommendation pipeline. They produce reports, plots, or console summaries but do not
modify runtime assets.

Examples:
- Anchor coverage and grid sweeps
- Pool size and fallback diagnostics
- Plain-color neutrality and constraint sanity checks

## Experiments

`experiments/` contains exploratory or one-off analyses used during model development.
These scripts are not guaranteed to be stable and may depend on intermediate data artifacts.

## Tools

`tools/` contains small utilities and helper scripts (e.g. environment setup, data inspection).

---

## Usage notes

Most scripts expect environment variables pointing to data files (inventory CSVs, caches, etc.).
See `tools/env.example.sh` for a reference configuration.
