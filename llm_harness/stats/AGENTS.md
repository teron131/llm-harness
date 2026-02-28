# Codemap: `llm_harness/stats`

## Scope

- Python mirror of the TypeScript stats pipeline for model source fetching, cross-source matching, and final selected output.

## What Module Is For

- Keep data source clients API-only (`artificial_analysis`, `models_dev`).
- Keep matching/scoring logic in `data_sources/matcher.py`.
- Keep final selected payload and cache behavior in `model_stats.py`.

## High-signal locations

- `llm_harness/stats/data_sources/artificial_analysis.py -> Artificial Analysis fetch/rank/enrich`
- `llm_harness/stats/data_sources/models_dev.py -> models.dev fetch/flatten/rank`
- `llm_harness/stats/data_sources/matcher.py -> cross-source matching + union/mapping outputs`
- `llm_harness/stats/model_stats.py -> final selected payload, list-cache behavior`
- `llm_harness/stats/__init__.py -> package export surface`

## Project-specific conventions and rationale

- Preserve payload keys/schema parity with the TS implementation.
- Public APIs are failure-safe by design and return `None`/empty payloads on errors.
- Data sources are API-style (no raw/source file outputs).
- Final list output in `model_stats.py` is cache-first (< 1 day), while single-id mode is in-memory only.

## Validation commands

- `uv run ruff check llm_harness/stats`
- `uv run ruff format llm_harness/stats`
- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
