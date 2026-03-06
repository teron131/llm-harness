# Codemap: `llm_harness/stats/image`

## Scope

- Python image stats pipeline and source/matcher entrypoints.

## What Module Is For

- `image_stats.py` is the public image stats entrypoint.
- `image_stats/` contains the staged pipeline that builds the final selected payload.
- `matcher.py` is the public image matcher API.
- `sources/` contains the upstream Artificial Analysis and Arena adapters.
