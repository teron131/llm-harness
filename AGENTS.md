## Section map

- Priority rules (mandatory behavior)
- Quick operating loop (default execution flow)
- Agent harness tools (tool choice and tradeoffs)
- Core principles (code quality and style)
- Standards & types (language defaults)
- Environment & tooling (setup, lint, build)
- Advanced topics (error handling, concurrency, ops)

---

## Priority rules (follow these)

1. **Repository awareness first**: inspect real files before proposing edits.
2. **Tooling defaults**: use `rg --files` for inventory, `rg` for text/symbol search, and `ast-grep_*` for syntax-aware structural edits.
3. **Respect conventions and rationale**: preserve documented schema design and caching boundaries unless explicitly asked to change them.

---

## Codemap Snapshot

# Codemap Report

## Scope

- Path: `llm-harness`
- Target focus: none
- Stack detected: Python

## High-Signal Locations

- `update_deps.py -> Ensure lockfile and environment are up to date.` - match for `if __name__ == ['\"']__main__['\"']`

## Project-Specific Conventions To Notice

- `llm_harness/__init__.py -> module` - from functools import cache
- `llm_harness/tools/youtube/summarizer.py -> module` - from .schemas import (
- `llm_harness/tools/youtube/summarizer.py -> SummarizerState` - """State schema for the summarization graph."""
- `llm_harness/tools/youtube/summarizer.py -> SummarizerOutput` - """Output schema for the summarization graph."""
- `llm_harness/tools/youtube/summarizer.py -> create_graph` - output_schema=SummarizerOutput,
- `llm_harness/tools/youtube/prompts.py -> get_langchain_summary_prompt` - "- Ensure output matches the provided response schema",
- `llm_harness/tools/youtube/prompts.py -> get_quality_check_prompt` - "For each aspect in the response schema, return a rating (Fail/Refine/Pass) and a specific, actionable reason.\n"
- `llm_harness/tools/youtube/summarizer_gemini.py -> module` - from .schemas import Summary

## Search Strategy (Syntax-First)

- Syntax relationship mapping and structural analysis: `ast-grep_*`
- Text and file discovery for narrowing: `rg --files`, `rg`
- Escalate to other tools only when these are not suitable

## Tackle Checklist

1. Start with `ast-grep_*` to map syntax-level connections across the target scope.
2. Verify invariants and caching/schema rationale before changing behavior.
3. Use `rg` to quickly narrow search windows and validate coverage.
4. Run validation commands for the touched scope.

## Validation Commands

- Python lint/format: `uv run ruff check . --fix && uv run ruff format .`

## Package Scripts

- No npm scripts discovered in scope.

## Optional AGENTS.md Export

- Export this map as AGENTS-style docs:
- `uv run python skills/codemap/scripts/generate_agents_md.py --repo . --mode targeted --targets backend fastapi --write --output AGENTS.md`
