# Codemap: `llm-harness-js/src/clients`

## Scope

- Provider client initialization, multimodal formatting, parser helpers, and usage tracking.

## What Module Is For

- This module centralizes TypeScript provider adapters, parsing, multimodal payload shaping, and usage tracking.

## High-signal locations

- `src/clients/openrouter.ts -> ChatOpenRouter/OpenRouterEmbeddings`
- `src/clients/gemini.ts -> ChatGemini/GeminiEmbeddings/createGeminiCache`
- `src/clients/multimodal.ts -> MediaMessage`
- `src/clients/parser.ts -> parseInvoke/parseBatch/parseStream/getMetadata`
- `src/clients/usage.ts -> UsageMetadata + context-local tracking`
- `src/clients/index.ts -> public re-export surface`

## Repository snapshot

- High-signal files listed below form the stable architecture anchors for this module.
- Keep imports and exports aligned with these anchors when extending behavior.

## Symbol Inventory

- Primary symbols are enumerated in the high-signal locations and syntax relationship sections.
- Preserve existing exported names unless changing a public contract intentionally.

## Key takeaways per location

- `openrouter.ts` applies routing/plugin/provider defaults and model-format validation.
- `gemini.ts` resolves key env vars and wraps cache upload/polling behavior.
- `multimodal.ts` converts local bytes/files into OpenAI-style content blocks.
- `parser.ts` unifies output/usage extraction across provider response shapes.
- `usage.ts` accumulates usage per async context via `AsyncLocalStorage`.

## Project-specific conventions and rationale

- Keep env var names stable (`OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`).
- Preserve parser fallback order to avoid provider-specific regressions.
- Keep provider-specific request logic inside client modules, not agent modules.

## Syntax Relationships

- `openrouter.ts -> import NativeChatOpenRouter from @langchain/openrouter`
- `gemini.ts -> import GoogleGenAI + ChatGoogleGenerativeAI`
- `multimodal.ts -> MediaMessage used by agents/ImageAnalysisAgent`
- `parser.ts -> streamed chunk parsing path + optional reasoning extraction`
- `usage.ts -> trackUsage used by agents/youtube/summarizerGemini.ts`

## General approach (not rigid checklist)

- Add new provider adapters as sibling modules and re-export via `clients/index.ts`.
- Keep schemas/outputs normalized before exposing to agents.
- Treat `usage.ts` as the only source of truth for accumulated costs/tokens.

## Validation commands

- `cd llm-harness-js && npm run build`
- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
