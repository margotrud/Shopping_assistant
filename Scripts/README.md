
# Scripts

Utility and one-off scripts used to build assets, run QA analyses,
and debug intermediate steps of the pipeline.

## Structure
- `build_*`: asset generation (lexicons, anchors, enriched datasets)
- `colors/`: color-specific build utilities
- `qa_*`: coverage and quality analysis scripts
- `debug_*`: exploratory or diagnostic scripts

## Notes
These scripts are not part of the public API and may rely on
local environment variables or large input files not tracked in Git.
