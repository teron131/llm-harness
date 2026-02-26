# Codemap: `llm-harness-js/src/tools`

## Scope

- Side-effect boundaries for filesystem operations, web loading, and YouTube transcript acquisition.

## High-signal locations

- `src/tools/index.ts -> top-level tool exports`
- `src/tools/fs/fsTools.ts -> sandboxed file primitives`
- `src/tools/fs/fastCopy.ts -> line tag/filter utilities`
- `src/tools/web/webloader.ts -> URL-to-markdown converter`
- `src/tools/youtube/scraper.ts -> transcript provider fallback + normalization`

## Key takeaways per location

- `index.ts` defines the tool surface consumed by agents.
- `fsTools.ts` is the root-dir trust boundary and must stay strict about path traversal.
- `fastCopy.ts` powers transcript garbage filtering via `[Lx]` range tags.
- `webloader.ts` intentionally returns `null` per failed URL instead of failing batch calls.
- `youtube/scraper.ts` encapsulates provider fallback (ScrapeCreators first, Supadata second).

## Project-specific conventions and rationale

- Keep side effects in `tools/*` and keep orchestration in `agents/*`.
- Preserve provider fallback ordering and user-facing errors for transcript fetching.
- Keep file operations constrained to sandbox root.

## Syntax relationship highlights (ast-grep-first)

- `tools/index.ts -> exports fs/web/youtube tool APIs`
- `agents/youtube/* -> import getTranscript + fastCopy helpers`
- `youtube/scraper.ts -> imports youtube utils for URL/text normalization`

## General approach (not rigid checklist)

- Add new external integrations as sibling tool modules.
- Keep return shapes stable across tool wrappers.
- Normalize provider responses before exposing them outside `tools/youtube`.

## Validation commands

- `cd llm-harness-js && npm run build`
- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
