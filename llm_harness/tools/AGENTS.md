# Codemap: `llm_harness/tools`

## Scope

- Tool-callable integration boundaries: filesystem editing, web loading, and YouTube transcript retrieval.

## High-signal locations

- `llm_harness/tools/__init__.py -> package re-exports`
- `llm_harness/tools/fs/fs_tools.py -> make_fs_tools`
- `llm_harness/tools/fs/fast_copy.py -> TagRange/tag_content/filter_content/untag_content`
- `llm_harness/tools/web/webloader.py -> webloader/webloader_tool`
- `llm_harness/tools/youtube/scraper.py -> scrape_youtube/get_transcript`

## Key takeaways per location

- `tools/__init__.py` is the consolidated import surface consumed by agents and downstream callers.
- `fs/fs_tools.py` enforces sandbox root + traversal guards for file operations.
- `fs/fast_copy.py` enables line-tag based filtering workflow used by YouTube summarizers.
- `web/webloader.py` converts URLs to markdown concurrently and sanitizes noisy artifacts.
- `youtube/scraper.py` abstracts transcript provider fallback and normalizes result shape.

## Project-specific conventions and rationale

- Keep side-effecting operations in tools, not in core agent orchestration.
- Preserve safety boundaries in filesystem tools (`SandboxFS.resolve` invariants).
- Keep transcript acquisition provider-agnostic via fallback logic (ScrapeCreators -> Supadata).
- Preserve tool-friendly function signatures and return types for LangChain integration.

## Syntax relationship highlights (ast-grep-first)

- `agents/youtube/summarizer.py -> tools.fs.fast_copy + tools.youtube.scraper`
- `agents/__init__.py -> tools.web.webloader_tool`
- `tools/__init__.py -> re-exports fs/web/youtube symbols`
- `tools/web/webloader.py -> @tool webloader_tool -> webloader`

## General approach (not rigid checklist)

- Extend tool behavior behind existing exported functions before adding new entrypoints.
- Keep tool modules domain-focused (`fs`, `web`, `youtube`) and avoid cross-domain coupling.
- When adding a tool, export it in `tools/__init__.py` only after interface is stable.

## Validation commands

- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
- `uv run ruff check .`
