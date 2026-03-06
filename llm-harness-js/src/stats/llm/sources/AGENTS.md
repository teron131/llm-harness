# Codemap: `llm-harness-js/src/stats/llm/sources`

## Scope

- Upstream adapters for LLM stats source data.

## What Module Is For

- `artificial-analysis-api.ts -> direct AA API fetch and score/rank helpers`
- `artificial-analysis-scraper.ts -> scraper fallback and evaluation/intelligence extraction`
- `models-dev.ts -> models.dev provider/model catalog`
- `openrouter-scraper.ts -> OpenRouter speed/pricing enrichment`

## Project-specific conventions and rationale

- Keep source adapters failure-safe and self-contained.
- The scraper path is the primary fallback-safe source for final LLM stats.
- The AA API adds richer fields when available but should not replace scraper-preferred fields wholesale.
- Source adapters should normalize upstream shapes, not decide final payload structure.
