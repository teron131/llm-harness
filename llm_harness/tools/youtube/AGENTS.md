# Codemap: `llm_harness/tools/youtube`

## Scope

- Transcript acquisition boundary for YouTube URLs with provider fallback and normalized result schema.

## What Module Is For

- This module fetches YouTube transcripts with provider fallback and normalized output shapes.

## High-signal locations

- `llm_harness/tools/youtube/scraper.py -> YouTubeScrapperResult`
- `llm_harness/tools/youtube/scraper.py -> _fetch_scrape_creators`
- `llm_harness/tools/youtube/scraper.py -> _fetch_supadata`
- `llm_harness/tools/youtube/scraper.py -> scrape_youtube/get_transcript`
- `llm_harness/tools/youtube/__init__.py -> public exports`

## Repository snapshot

- High-signal files listed below form the stable architecture anchors for this module.
- Keep imports and exports aligned with these anchors when extending behavior.

## Symbol Inventory

- Primary symbols are enumerated in the high-signal locations and syntax relationship sections.
- Preserve existing exported names unless changing a public contract intentionally.

## Key takeaways per location

- `YouTubeScrapperResult` is the normalization layer consumed by agent workflows.
- `_fetch_scrape_creators` and `_fetch_supadata` isolate provider-specific API calls and parsing.
- `scrape_youtube` controls fallback ordering and user-facing error semantics.
- `get_transcript` converts provider output into cleaned transcript text with strict validation.

## Project-specific conventions and rationale

- Preserve provider fallback order (ScrapeCreators first, Supadata second).
- Keep missing/invalid provider responses soft (`None`) until top-level `scrape_youtube` decides failure.
- Keep transcript cleanliness and URL normalization via shared utilities in `utils.youtube_utils`.
- Environment keys are optional per provider, but at least one provider key is required for success.

## Syntax Relationships

- `scraper.py -> scrape_youtube -> _fetch_scrape_creators/_fetch_supadata`
- `scraper.py -> YouTubeScrapperResult.parsed_transcript -> clean_text`
- `agents/youtube/summarizer.py -> get_transcript`
- `agents/youtube/__init__.py -> youtube_loader -> scrape_youtube`

## General approach (not rigid checklist)

- For new transcript providers, add isolated fetch helpers and plug them into `scrape_youtube` fallback chain.
- Keep `YouTubeScrapperResult` as the canonical cross-provider structure.
- Avoid leaking raw provider payload shapes beyond this module.

## Validation commands

- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
- `uv run ruff check .`
