# Codemap: `llm-harness` (repo root)

## Project Summary

- Python-only LLM harness package with agents, provider clients, tools, and stats helpers.

## Structure and entrypoints

- `llm_harness -> Python package root`
- `llm_harness/agents -> orchestration layer`
- `llm_harness/clients -> provider adapters + parsing + usage tracking`
- `llm_harness/tools -> side-effect boundaries (web, fs, youtube)`
- `llm_harness/utils -> pure helper transforms`
- `pyproject.toml -> Python build/lint config`

## Core flows and rationale

- Agent workflows should remain split by concern:
  - orchestration in `agents`
  - provider/message/usage mechanics in `clients`
  - network/filesystem integration in `tools`
  - deterministic transforms in `utils`
- The sibling repo `../llm-harness-js` is the TypeScript port when parity work is needed.
- Preserve schema-driven outputs in YouTube summarization paths to keep downstream handling predictable.

## Always-on rules

- Keep model names stable unless code references require updates.
- Prefer extending existing module boundaries instead of introducing cross-layer shortcuts.
- Add or adjust module-level `AGENTS.md` when introducing new source submodules.
- Run required verification after edits:
  - `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`

## Repository snapshot

- Source modules with `AGENTS.md` now cover:
  - Python: `llm_harness`, `agents`, `agents/youtube`, `clients`, `tools`, `tools/fs`, `tools/web`, `tools/youtube`, `utils`
- Structural signals (from module stats):
  - Python import edges: 160
