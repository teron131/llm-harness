# Codemap: `llm_harness`

## Scope

- This guide maps package-level responsibilities and cross-module boundaries inside `llm_harness`.

## High-signal locations

- `llm_harness/agents -> module`
- `llm_harness/clients -> module`
- `llm_harness/tools -> module`
- `llm_harness/utils -> module`

## Key takeaways per location

- `llm_harness/agents -> module`: Orchestrates LLM workflows and exposes preconfigured agent entrypoints.
- `llm_harness/clients -> module`: Holds model client adapters (`openrouter`, `gemini`), multimodal message construction, response parsing, and usage accumulation.
- `llm_harness/tools -> module`: Tool-callable boundaries for web loading, filesystem operations, and YouTube transcript fetching.
- `llm_harness/utils -> module`: Pure helper layer for text conversion, image encode/display, and YouTube URL/text normalization.

## Project-specific conventions and rationale

- Keep model routing in `clients/*`; workflow logic stays in `agents/*`.
- Keep I/O boundaries in `tools/*`; avoid duplicating API calls inside agents when a tool already exists.
- Keep pure transforms in `utils/*` to make pipeline behavior testable and composable.
- Preserve schema-driven outputs in YouTube flows (`Summary`, `Quality`, `GarbageIdentification`) to maintain predictable downstream behavior.

## Syntax relationship highlights (ast-grep-first)

- `llm_harness/agents/agents.py -> BaseHarnessAgent -> ChatOpenRouter`
- `llm_harness/agents/youtube/summarizer.py -> get_transcript/filter_content/tag_content/untag_content`
- `llm_harness/agents/youtube/summarizer_gemini.py -> track_usage`
- `llm_harness/tools/__init__.py -> re-exports fs/web/youtube tools`
- `llm_harness/clients/__init__.py -> re-exports client/runtime primitives + agent classes`

## General approach (not rigid checklist)

- Start from package `__init__` exports to identify supported public surface.
- Trace one execution path end-to-end before refactoring shared code:
  YouTube input -> tools -> agent workflow -> schema output -> usage tracking.
- When changing behavior, enforce boundaries:
  agent orchestration in `agents`, provider mechanics in `clients`, side effects in `tools`.

## Validation commands

- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
- `uv run ruff check .`
