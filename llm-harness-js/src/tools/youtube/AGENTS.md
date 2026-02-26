# Codemap: `llm-harness-js/src/tools/youtube`

## Scope

- YouTube transcript scraping boundary with provider fallback and normalized result schema.

## High-signal locations

- `src/tools/youtube/scraper.ts -> scrapeYoutube/getTranscript/formatYoutubeLoaderOutput`
- `src/tools/youtube/index.ts -> re-export surface`

## Key takeaways per location

- `scrapeYoutube` validates URL, normalizes it, then tries providers in fixed order.
- `_fetch_scrape_creators` equivalent runs first; Supadata fallback runs second.
- `getTranscript` enforces strict transcript availability and non-empty output.
- `formatYoutubeLoaderOutput` converts normalized result into a readable block used by agent APIs.

## Project-specific conventions and rationale

- Keep provider fallback order deterministic to match existing behavior.
- Keep env key handling soft per provider but strict overall when none are configured.
- Keep transcript normalization centralized in this module.

## Syntax relationship highlights (ast-grep-first)

- `scraper.ts -> imports cleanText/cleanYoutubeUrl/extractVideoId/isYoutubeUrl`
- `agents/youtube/index.ts -> youtubeLoader uses scrapeYoutube + formatYoutubeLoaderOutput`
- `agents/youtube/summarizer*.ts -> getTranscript for transcript source`

## General approach (not rigid checklist)

- Add new providers as isolated fetch helpers and plug into `scrapeYoutube`.
- Keep normalized result shape stable before exposing to agents.
- Keep user-facing error semantics explicit and actionable.

## Validation commands

- `cd llm-harness-js && npm run build`
- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
