# Codemap: `llm-harness-js/src/stats`

## Scope

- Cross-source stats fetchers, matchers, and final projection helpers for LLM and image model analytics.

## What Module Is For

- `index.ts` is the public export surface for stats helpers.
- `llm/` owns the LLM stats pipeline:
  - source fetch,
  - cross-source matching,
  - OpenRouter enrichment,
  - final payload projection.
- `image/` owns the image leaderboard fetch/projection path:
  - staged final payload pipeline in `image-stats/`
  - AA/Arena source adapters in `sources/`
  - Arena matching heuristics in `matcher.ts`
- Keep raw source adapters, matcher logic, and final selected payload builders in separate modules.

## High-signal locations

- `llm/llm-stats.ts -> public LLM stats API and cache-first orchestration`
- `llm/llm-stats/ -> staged LLM stats pipeline (source -> match -> OpenRouter enrich -> final build)`
- `llm/matcher.ts -> public matcher APIs and scraper fallback diagnostics`
- `llm/matcher/ -> provider scoping, candidate scoring, tokenizer helpers, source-model builders`
- `llm/sources/ -> upstream Artificial Analysis, models.dev, and OpenRouter adapters`
- `image/image-stats.ts -> public image stats API and cache-first orchestration`
- `image/image-stats/ -> staged image stats pipeline (source -> match -> final build)`
- `image/sources/ -> upstream Artificial Analysis and Arena image adapters`

## Project-specific conventions and rationale

- Keep top-level output keys stable for `.cache/*.json` interoperability.
- Prefer scraper-first LLM stats rows, then use the Artificial Analysis API as field-level fallback when it adds missing evaluations.
- Keep OpenRouter enrichment separate from source fetch/matching because it is a later fallback layer.
- Keep raw `scores` formula-native and expose normalized user-facing ranking values separately.
- When adding benchmarks or fields, preserve scraper precedence unless a field is missing.
