# Codemap: `llm-harness-js/src/tools/web`

## Scope

- URL content loading and markdown cleanup for agent/tool consumption.

## What Module Is For

- This module converts URLs into markdown in batch and exposes the web loader tool surface.

## High-signal locations

- `src/tools/web/webloader.ts -> webloader/webloaderTool`
- `src/tools/web/index.ts -> module re-exports`

## Repository snapshot

- High-signal files listed below form the stable architecture anchors for this module.
- Keep imports and exports aligned with these anchors when extending behavior.

## Symbol Inventory

- Primary symbols are enumerated in the high-signal locations and syntax relationship sections.
- Preserve existing exported names unless changing a public contract intentionally.

## Key takeaways per location

- `webloader` converts one-or-many URLs to markdown and preserves output ordering.
- Conversion errors are isolated as `null` entries, not global failures.
- Cleanup step removes noisy artifacts and excessive whitespace.

## Project-specific conventions and rationale

- Keep loader resilient for batch workloads (partial success is expected).
- Keep cleanup conservative to avoid changing source meaning.
- Maintain the `webloaderTool` wrapper for orchestration compatibility.

## Syntax Relationships

- `agents/index.ts -> getTools() includes webloaderTool`
- `agents/agents.ts -> WebLoaderAgent/WebSearchLoaderAgent use webloaderTool`

## General approach (not rigid checklist)

- Add behavior in `webloader` first, then expose through `webloaderTool`.
- Preserve return type contract: `Array<string | null>`.

## Validation commands

- `cd llm-harness-js && npm run build`
- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
