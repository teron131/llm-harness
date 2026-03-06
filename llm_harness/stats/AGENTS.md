# Codemap: `llm_harness/stats`

## Scope

- Python mirror of the TypeScript stats pipeline for model source fetching, cross-source matching, and final selected output.

## What Module Is For

- Keep the Python tree structurally aligned with `llm-harness-js/src/stats`.
- Keep LLM source adapters in `llm/sources/`, matcher internals in `llm/matcher/`, public entrypoint in `llm/llm_stats.py`, and staged orchestration in `llm/llm_stats_stages/`.
- Keep image source adapters in `image/sources/`, matcher logic in `image/matcher.py`, and staged orchestration in `image/image_stats/`.
- Keep cross-cutting helpers in `utils.py` and LLM-specific helpers in `llm/shared.py`.

## High-signal locations

- `llm_harness/stats/utils.py -> shared numeric/cache/http helpers mirrored from JS`
- `llm_harness/stats/llm/sources/*.py -> LLM source fetching + normalization`
- `llm_harness/stats/llm/matcher/*.py -> scraper/API to models.dev matching`
- `llm_harness/stats/llm/llm_stats.py -> public LLM stats entrypoint`
- `llm_harness/stats/llm/llm_stats_stages/*.py -> staged LLM cache/source/match/openrouter/final flow`
- `llm_harness/stats/image/sources/*.py -> image source fetching + normalization`
- `llm_harness/stats/image/matcher.py -> image cross-source matching`
- `llm_harness/stats/image/image_stats/*.py -> staged image cache/source/match/final flow`
- `llm_harness/stats/__init__.py -> package export surface`

## Project-specific conventions and rationale

- Preserve payload keys/schema parity with the TS implementation.
- Public APIs are failure-safe by design and return `None`/empty payloads on errors.
- Final list outputs in `llm/llm_stats.py` and `image/image_stats/` are cache-first (< 1 day), while single-id mode is in-memory only.

## Validation commands

- `uv run ruff check llm_harness/stats`
- `uv run ruff format llm_harness/stats`
- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
