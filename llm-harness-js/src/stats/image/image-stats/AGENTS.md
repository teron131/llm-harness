# Codemap: `llm-harness-js/src/stats/image/image-stats`

## Scope

- Final selected image stats payload pipeline.

## Pipeline

- `source-stage.ts`
  - fetch Artificial Analysis and Arena payloads
  - build lookup maps keyed by slug or Arena model name
- `match-stage.ts`
  - run the image matcher against the fetched source rows
  - attach AA and Arena source rows to matcher output
  - append unmatched Arena rows as image-only fallbacks
- `final-stage.ts`
  - project matched rows into the final public model shape
  - compute averaged scores and percentiles
  - sort and exact-id filter
- `cache.ts`
  - list-mode cache read/write only
- `types.ts`
  - public payload types and stage handoff types

## Project-specific conventions and rationale

- Keep the public payload shape stable.
- Keep list mode cache-first; single-model mode stays in-memory only.
- Treat Artificial Analysis and Arena as peer sources, then average aligned score fields in the final stage.
