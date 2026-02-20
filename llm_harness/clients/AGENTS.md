# Codemap: `llm_harness/clients`

## Scope

- Provider clients, multimodal message construction, response parsing, and usage tracking.

## High-signal locations

- `llm_harness/clients/openrouter.py -> ChatOpenRouter/OpenRouterEmbeddings`
- `llm_harness/clients/gemini.py -> ChatGemini/GeminiEmbeddings/create_gemini_cache`
- `llm_harness/clients/multimodal.py -> MediaMessage`
- `llm_harness/clients/parser.py -> StructuredOutput/parse_*`
- `llm_harness/clients/usage.py -> UsageMetadata/track_usage/get_usage`
- `llm_harness/clients/__init__.py -> package surface`

## Key takeaways per location

- `openrouter.py` is OpenAI-compatible transport setup with plugin routing (web, pdf parser).
- `gemini.py` provides Gemini-native setup and cache upload flow with file processing polling.
- `multimodal.py` normalizes paths/bytes to OpenAI Chat Completions content blocks.
- `parser.py` unifies metadata extraction across provider-specific response shapes and stream/non-stream parsing.
- `usage.py` tracks per-execution token/cost totals via `ContextVar`.

## Project-specific conventions and rationale

- Keep provider-specific initialization isolated by file (`openrouter.py`, `gemini.py`).
- Preserve schema-first parsing (`StructuredOutput`) and avoid ad-hoc response dict access in call sites.
- Usage totals are context-local; workflows should reset usage at run start.
- `MediaMessage` is the canonical path for multimodal content into chat models.

## Syntax relationship highlights (ast-grep-first)

- `agents/agents.py -> BaseHarnessAgent -> ChatOpenRouter`
- `agents/youtube/summarizer_gemini.py -> track_usage`
- `clients/__init__.py -> re-export parser/usage/provider APIs`
- `clients/multimodal.py -> MediaMessage(HumanMessage) -> content block builders`

## General approach (not rigid checklist)

- Add new provider clients as separate files, then expose via `clients/__init__.py`.
- Keep output normalization in `parser.py` instead of spreading parsing branches across callers.
- Treat `usage.py` as the single source of truth for aggregated usage/cost state.

## Validation commands

- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
- `uv run ruff check .`
