# Codemap: `llm-harness-js/src/stats`

## Scope

- Cross-source model analytics, matching, and projection utilities built from provider stats feeds.

## What Module Is For

- Keep data-source fetch/cache logic separate from matching/scoring logic.
- Expose reusable APIs for:
  - fetching normalized Artificial Analysis stats,
  - fetching normalized models.dev stats,
  - matching and unioning models (OpenRouter-only),
  - projecting final selected payloads for downstream consumers.

## High-signal locations

- `src/stats/data-sources/artificialAnalysis.ts -> Artificial Analysis fetch/cache/filter/rank pipeline`
- `src/stats/data-sources/modelsDev.ts -> models.dev fetch/cache/flatten/filter pipeline`
- `src/stats/data-sources/matcher.ts -> cross-source matching, candidate scoring, union generation, and void-thresholding`
- `src/stats/modelStats.ts -> final selected projection used by test/output scripts`
- `src/stats/index.ts -> public re-export surface for stats helpers`

## Project-specific conventions and rationale

- Keep output payload keys stable for `.cache/*` JSON interoperability.
- Keep fetch/cache behavior deterministic; use explicit options for refresh/TTL.
- Keep scoring constants explicit and named as rewards/penalties when adding new matching logic.
- Keep matcher provider scope fixed to OpenRouter for deterministic results.
- Keep source-of-truth names from models.dev in merged outputs when available.
