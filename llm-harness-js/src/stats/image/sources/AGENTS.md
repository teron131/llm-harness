# Codemap: `llm-harness-js/src/stats/image/sources`

## Scope

- Upstream image leaderboard/source adapters.

## What Module Is For

- `artificial-analysis.ts -> fetch and enrich Artificial Analysis text-to-image data`
- `arena-ai.ts -> scrape and aggregate Arena text-to-image leaderboard categories`

## Project-specific conventions and rationale

- Keep source adapters failure-safe and return empty-but-shaped payloads where the public API expects them.
- Keep source-specific taxonomy/grouping logic inside the source module, not in later pipeline stages.
