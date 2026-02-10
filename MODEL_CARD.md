# Model Card — Lipstick Recommender (v1)

## Objective
Recommend lipstick shades from free-text preferences using an interpretable, debuggable pipeline.

## Inputs
- User text (English short queries)
- Inventory with shade chips + Lab/HSL-derived features

## Outputs
- Ranked list of shades with scoring breakdown (ΔE to anchor, constraint penalty/tilt, total score)

## What is modeled
- Color intent extraction (mentions)
- Anchor resolution (lexicon + optional domain anchor)
- Candidate pool selection + Lab-based scoring
- Axis-style constraints where supported (e.g., brightness/depth via constraints)

## What is not modeled (v1)
- Full semantic coverage of cosmetic finish (e.g., glossy, shimmer, creamy)
- Warm/cool as a dedicated axis influencing scoring (may be detected but not propagated)
- Ground-truth “color family” classification labels (evaluation uses proxies)

## Evaluation
Offline evaluation uses proxy metrics derived from Lab space:
- Hue-family proxy precision (macro hue classes with chroma gate)
- Neutral proxy precision (Lab lightness/chroma + depth heuristics)
- Latency p50/p95

## Known limitations (intentional trade-offs)
- Neutral families (nude/peach/beige/brown) are handled via Lab heuristics rather than explicit family labels.
- Some parsed constraints are not yet propagated end-to-end to scoring.
- The system prioritizes stability + interpretability over exhaustive NLP coverage.

## Next steps
- Propagate warm/cool signal into scoring (or explicitly drop it from NLP).
- Replace proxy metrics with a small labeled eval set.
- Add per-component ablations (anchor-only vs constraints-only vs full pipeline).
