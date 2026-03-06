# Codemap: `llm-harness-js/src/stats/llm`

## Scope

- LLM-specific source adapters, matcher utilities, and final selected stats payload builders.

## What Module Is For

- `llm-stats.ts` is the public LLM stats entrypoint.
- `llm-stats/` contains the staged pipeline that builds the final selected payload.
- `matcher.ts` is the public API for cross-source LLM model matching and diagnostics.
- `matcher/` contains the lower-level matcher pipeline, scoring heuristics, tokenization, and source-model builders.
- `sources/` contains the upstream data-source adapters for Artificial Analysis, models.dev, and OpenRouter.

## High-signal locations

- `llm-stats.ts -> fetch/cache orchestration for final selected model stats`
- `llm-stats/source-stage.ts -> scraper + API source fetch and lookup maps`
- `llm-stats/match-stage.ts -> scraper-first matched rows with API fallback merge`
- `llm-stats/openrouter-stage.ts -> OpenRouter enrichment and row-level fallback policy`
- `llm-stats/final-stage.ts -> final model projection, relative scores, sort/prune`
- `matcher.ts -> public mapping and scraper fallback diagnostics`

## Project-specific conventions and rationale

- Scraper rows are the primary LLM stats source of truth; the AA API only fills missing fields.
- models.dev is the canonical provider/model identity source once a match is selected.
- OpenRouter enrichment is late-bound and should not replace earlier source fields outside speed/pricing fallback.
- Keep raw `scores` and normalized `relative_scores` distinct.
