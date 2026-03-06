# Codemap: `llm-harness-js/src/stats/image`

## Scope

- Image-model stats pipeline and public source/matcher entrypoints.

## What Module Is For

- `image-stats.ts` is the public image stats entrypoint.
- `image-stats/` contains the staged pipeline that builds the final selected payload.
- `matcher.ts` remains the public image matcher API.
- `sources/` contains the upstream Artificial Analysis and Arena adapters.

## High-signal locations

- `image-stats.ts -> public image stats API and cache-first orchestration`
- `image-stats/source-stage.ts -> source fetch and lookup-map construction`
- `image-stats/match-stage.ts -> matcher merge plus unmatched Arena fallback rows`
- `image-stats/final-stage.ts -> final model projection, score averaging, sort/filter`
- `matcher.ts -> AA-to-Arena matching heuristics and candidate ranking`
- `sources/ -> upstream image leaderboard adapters`
