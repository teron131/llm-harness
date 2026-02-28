# Codemap: `llm-harness-js/src/stats`

## Scope

- Cross-source model analytics, matching, and ranking utilities built from provider stats feeds.

## What Module Is For

- Keep scoring and merge logic separate from provider client/fetch logic.
- Expose reusable APIs that consume normalized stats payloads from `src/stats/evalStats.ts` and `src/stats/modelStats.ts`.

## High-signal locations

- `src/stats/evalStats.ts -> AA eval fetch/cache/filter/rank pipeline`
- `src/stats/modelStats.ts -> models.dev fetch/cache/flatten/filter pipeline`
- `src/stats/index.ts -> public re-export surface for stats helpers`

## Project-specific conventions and rationale

- Keep output payload keys stable for `.cache/*` JSON interoperability.
- Keep fetch/cache behavior deterministic and configurable via env knobs.
- Keep scoring constants explicit and named as rewards/penalties when adding new matching logic.

## Validation commands

- `cd llm-harness-js && npm run build`
- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
