# Codemap: `llm-harness-js/src/stats/llm/matcher`

## Scope

- Internal matcher pipeline used to map Artificial Analysis models to models.dev models.

## What Module Is For

- `pipeline.ts -> provider scoping, candidate collection, void-thresholding, final matcher run`
- `scoring.ts -> candidate scoring heuristics`
- `tokenize.ts -> model-id tokenization and numeric/B-scale parsing`
- `source-model.ts -> build matcher source rows from AA API or scraper rows`
- `types.ts -> matcher payloads and intermediate shapes`

## Project-specific conventions and rationale

- Prefer OpenRouter models when the primary pool produces a good match.
- Fall back to a small trusted provider set when needed.
- Keep token and scale heuristics explicit and additive so they are easy to tune.
- The matcher only picks candidate identities; final field merge happens in `llm-stats/match-stage.ts`.
