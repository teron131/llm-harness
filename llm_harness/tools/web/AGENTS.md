# Codemap: `llm_harness/tools/web`

## Scope

- URL-to-markdown loading utilities with lightweight cleaning and LangChain tool exposure.

## What Module Is For

- This module converts URLs to markdown content and exposes tool-callable web loading helpers.

## High-signal locations

- `llm_harness/tools/web/webloader.py -> _clean_markdown`
- `llm_harness/tools/web/webloader.py -> webloader`
- `llm_harness/tools/web/webloader.py -> webloader_tool`
- `llm_harness/tools/web/__init__.py -> re-export`

## Repository snapshot

- High-signal files listed below form the stable architecture anchors for this module.
- Keep imports and exports aligned with these anchors when extending behavior.

## Symbol Inventory

- Primary symbols are enumerated in the high-signal locations and syntax relationship sections.
- Preserve existing exported names unless changing a public contract intentionally.

## Key takeaways per location

- `webloader` is the core path: uses `DocumentConverter`, parallel conversion, and returns one output per input URL.
- `_clean_markdown` removes noise such as image comments and excessive whitespace.
- `webloader_tool` is the LangChain-facing wrapper used by agents.

## Project-specific conventions and rationale

- Conversion failures are intentionally non-fatal (`None` per URL) to keep batch behavior resilient.
- Threading is bounded (`min(len(urls), cpu_count, 10)`) to avoid excessive worker creation.
- Keep markdown cleanup conservative; avoid content-changing rewrites.

## Syntax Relationships

- `webloader.py -> webloader -> DocumentConverter.convert(...).document.export_to_markdown()`
- `webloader.py -> ThreadPoolExecutor -> executor.map(_convert, urls)`
- `agents/agents.py -> WebLoaderAgent/WebSearchLoaderAgent -> webloader_tool`

## General approach (not rigid checklist)

- Adjust cleaning rules in `_clean_markdown` before changing converter behavior.
- Keep tool interface stable (`list[str] -> list[str | None]` and wrapper type expectations).
- Prefer additive handling for new URL/document edge cases.

## Validation commands

- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
- `uv run ruff check .`
