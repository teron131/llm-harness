# Codemap: `llm-harness-js/src`

## Scope

- Source root for the TypeScript implementation; this module defines the public API and coordinates `agents`, `clients`, `tools`, and `utils`.

## What Module Is For

- `src` exists to provide a coherent, typed TS surface mirroring the Python harness behavior with idiomatic JavaScript/TypeScript internals.

## High-signal locations

- `src/index.ts -> export barrel for package consumers`
- `src/agents -> orchestration classes and YouTube summarization workflows`
- `src/clients -> model adapters, multimodal payloads, parsing, usage accounting`
- `src/tools -> integration boundaries (filesystem, web loading, youtube scraping)`
- `src/utils -> deterministic string/image/youtube helpers`

## Repository snapshot

- `src` contains 5 top-level source groups: `agents`, `clients`, `tools`, `utils`, and root exports/types.
- High-use local import targets include `./schemas.js`, `./prompts.js`, `../../utils/youtubeUtils.js`, and tool modules in `tools/*`.

## Symbol Inventory

- Entrypoint exports:
  - `src/index.ts -> re-exports agents, clients, tools, utils`
- Notable orchestrators:
  - `agents/agents.ts -> BaseHarnessAgent`, web/image agents, YouTube summarizer wrappers
  - `agents/youtube/summarizer*.ts -> summarization graph + quality/garbage analysis nodes`
- Notable clients/tools:
  - `clients/openrouter.ts`, `clients/gemini.ts`, `clients/multimodal.ts`, `clients/usage.ts`
  - `tools/youtube/scraper.ts`, `tools/web/webloader.ts`, `tools/fs/fastCopy.ts`

## Syntax Relationships

- `src/index.ts -> exports from ./agents/index.js`, `./clients/index.js`, `./tools/index.js`, `./utils/index.js`
- `agents/agents.ts -> imports ChatOpenRouter (clients/openrouter.ts) + MediaMessage (clients/multimodal.ts)`
- `agents/youtube/summarizer*.ts -> imports prompts/schemas + youtube scraper + fs copy helper`
- `tools/youtube/scraper.ts -> depends on utils/youtubeUtils.ts` for URL parsing and normalization
- `tools/*` and `clients/*` are dependencies of `agents/*`; `utils/*` are shared low-level helpers

## Key takeaways per location

- `src/index.ts`: treat as stable package contract for external callers.
- `src/agents`: compose clients/tools; avoid provider-specific request shaping here.
- `src/clients`: keep provider and usage mechanics centralized for consistent behavior.
- `src/tools`: isolate side effects and external API boundaries.
- `src/utils`: keep pure transforms reusable and side-effect free.

## Project-specific conventions and rationale

- Preserve naming parity with Python equivalents where classes and workflows correspond.
- Keep model defaults and environment variable semantics aligned with Python unless intentionally diverging.
- When adding new features, prefer extending existing submodule boundaries instead of creating new cross-cutting helpers.
- Required verification after edits:
  - `npm run build`
  - `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
