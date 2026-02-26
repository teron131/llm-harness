# Codemap: `llm-harness-js/src/tools/web`

## Scope

- URL content loading and markdown cleanup for agent/tool consumption.

## High-signal locations

- `src/tools/web/webloader.ts -> webloader/webloaderTool`
- `src/tools/web/index.ts -> module re-exports`

## Key takeaways per location

- `webloader` converts one-or-many URLs to markdown and preserves output ordering.
- Conversion errors are isolated as `null` entries, not global failures.
- Cleanup step removes noisy artifacts and excessive whitespace.

## Project-specific conventions and rationale

- Keep loader resilient for batch workloads (partial success is expected).
- Keep cleanup conservative to avoid changing source meaning.
- Maintain the `webloaderTool` wrapper for orchestration compatibility.

## Syntax relationship highlights (ast-grep-first)

- `agents/index.ts -> getTools() includes webloaderTool`
- `agents/agents.ts -> WebLoaderAgent/WebSearchLoaderAgent use webloaderTool`

## General approach (not rigid checklist)

- Add behavior in `webloader` first, then expose through `webloaderTool`.
- Preserve return type contract: `Array<string | null>`.

## Validation commands

- `cd llm-harness-js && npm run build`
- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
