# Codemap: `llm_harness/utils`

## Scope

- Pure helper functions for text conversion, image handling, and YouTube URL/text normalization.

## What Module Is For

- This module contains deterministic helper transforms for text conversion, image handling, and YouTube URL parsing.

## High-signal locations

- `llm_harness/utils/text_utils.py -> s2hk`
- `llm_harness/utils/youtube_utils.py -> clean_text/clean_youtube_url/is_youtube_url/extract_video_id`
- `llm_harness/utils/image_utils.py -> load_image_base64/display_image_base64`
- `llm_harness/utils/__init__.py -> re-exports`

## Repository snapshot

- High-signal files listed below form the stable architecture anchors for this module.
- Keep imports and exports aligned with these anchors when extending behavior.

## Symbol Inventory

- Primary symbols are enumerated in the high-signal locations and syntax relationship sections.
- Preserve existing exported names unless changing a public contract intentionally.

## Key takeaways per location

- `text_utils.s2hk` memoizes OpenCC converter initialization with `@cache`.
- `youtube_utils` centralizes URL identity and normalization rules reused by scraper and summarizers.
- `image_utils` loads from URL/local source, resizes proportionally, and emits base64 for multimodal use.

## Project-specific conventions and rationale

- Keep this layer side-effect light and deterministic.
- Use shared URL normalization from `youtube_utils` instead of duplicating regex logic in callers.
- Preserve conversion/helper behavior because these functions are consumed across tools and agents.
- Prefer utility reuse over adding one-off inline transforms in higher layers.

## Syntax Relationships

- `tools/youtube/scraper.py -> clean_text/clean_youtube_url/extract_video_id/is_youtube_url`
- `agents/youtube/schemas.py -> s2hk`
- `utils/__init__.py -> package-level utility re-exports`

## General approach (not rigid checklist)

- Add narrowly scoped helpers with clear names and explicit input/output behavior.
- Keep regex-driven behavior centralized here when shared by multiple modules.
- Avoid business orchestration logic in this layer.

## Validation commands

- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
- `uv run ruff check .`
