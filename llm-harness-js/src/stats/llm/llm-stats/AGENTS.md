# Codemap: `llm-harness-js/src/stats/llm/llm-stats`

## Scope

- Final selected LLM stats payload pipeline.

## Pipeline

- `source-stage.ts`
  - fetch scraper rows, Artificial Analysis API rows, and models.dev rows
  - build lookup maps keyed by slug or model id
- `match-stage.ts`
  - run scraper fallback diagnostics through the matcher
  - build matched rows
  - merge AA API fields into scraper rows as fallback only
- `openrouter-stage.ts`
  - dedupe rows with OpenRouter preference
  - backfill free-route costs
  - enrich rows with OpenRouter speed/pricing
- `final-stage.ts`
  - project matched rows into the final public model shape
  - compute raw `scores`
  - attach normalized `relative_scores`
  - sort, prune, and filter
- `cache.ts`
  - list-mode cache read/write only
- `types.ts`
  - public payload types and stage handoff types

## Project-specific conventions and rationale

- Keep the public payload shape stable.
- Keep list mode cache-first; single-model mode stays in-memory only.
- Scraper-first means:
  - keep scraper values when present
  - use the AA API only to fill holes such as missing evaluation keys
- Raw `scores` preserve the underlying formulas.
- `relative_scores` are the normalized comparison layer used for ranking and UI-like consumption.
