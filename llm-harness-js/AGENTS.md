# Codemap: `llm-harness-js`

## Scope

- TypeScript port of `llm_harness` with equivalent module boundaries: `agents`, `clients`, `tools`, `utils`.

## High-signal locations

- `llm-harness-js/src/index.ts -> package surface`
- `llm-harness-js/src/agents -> orchestration layer`
- `llm-harness-js/src/clients -> provider + parsing + usage primitives`
- `llm-harness-js/src/tools -> I/O and integration boundaries`
- `llm-harness-js/src/utils -> pure helper transforms`

## Key takeaways per location

- `src/index.ts` is the canonical export barrel; preserve this as the public entrypoint.
- `src/agents` composes clients and tools into task-level workflows.
- `src/clients` centralizes provider setup, multimodal message building, response parsing, and usage aggregation.
- `src/tools` contains side-effect boundaries (filesystem, web loading, YouTube scraping).
- `src/utils` keeps deterministic helpers isolated from orchestration logic.

## Project-specific conventions and rationale

- Preserve model defaults and env var names to maintain Python parity.
- Keep naming idiomatic in TS (`camelCase`) while retaining behavior contracts from Python.
- Keep provider-specific request shaping in `clients/*`, not in agents.
- Keep external API calls isolated to `tools/*`.

## Syntax relationship highlights (ast-grep-first)

- `src/index.ts -> export * from agents/clients/tools/utils`
- `src/agents/agents.ts -> ChatOpenRouter + MediaMessage + webloaderTool`
- `src/agents/youtube/summarizer*.ts -> tools/youtube/scraper + tools/fs/fastCopy + prompts/schemas`
- `src/clients/openrouter.ts -> @langchain/openrouter + @langchain/openai`
- `src/clients/gemini.ts -> @google/genai + @langchain/google-genai`
- `src/tools/youtube/scraper.ts -> utils/youtubeUtils.ts`

## General approach (not rigid checklist)

- Start at `src/index.ts` to identify the public API.
- Trace one user flow through `agents -> clients/tools -> schemas` before refactoring.
- Change behavior only where parity requires it; otherwise keep JS internals simple and explicit.

## Validation commands

- `cd llm-harness-js && npm run build`
- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
