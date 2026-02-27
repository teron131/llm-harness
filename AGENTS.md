# Codemap: `llm-harness` (repo root)

## Project Summary

- Dual-language LLM harness: `llm_harness` (Python package) and `llm-harness-js` (TypeScript port) expose parallel agent/client/tool utilities.

## Structure and entrypoints

- `llm_harness -> Python package root`
- `llm_harness/agents -> orchestration layer`
- `llm_harness/clients -> provider adapters + parsing + usage tracking`
- `llm_harness/tools -> side-effect boundaries (web, fs, youtube)`
- `llm_harness/utils -> pure helper transforms`
- `llm-harness-js/src -> TypeScript source root`
- `llm-harness-js/src/index.ts -> TS public export surface`
- `pyproject.toml -> Python build/lint config`
- `package.json -> root npm scripts/deps`

## Core flows and rationale

- Agent workflows should remain split by concern:
  - orchestration in `agents`
  - provider/message/usage mechanics in `clients`
  - network/filesystem integration in `tools`
  - deterministic transforms in `utils`
- Python and TypeScript implementations should stay behaviorally aligned where classes/flows share names.
- Preserve schema-driven outputs in YouTube summarization paths to keep downstream handling predictable.

## Always-on rules

- Keep model names stable unless code references require updates.
- Prefer extending existing module boundaries instead of introducing cross-layer shortcuts.
- Add or adjust module-level `AGENTS.md` when introducing new source submodules.
- Run required verification after edits:
  - `npm run build`
  - `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`

## Repository snapshot

- Source modules with `AGENTS.md` now cover:
  - Python: `llm_harness`, `agents`, `agents/youtube`, `clients`, `tools`, `tools/fs`, `tools/web`, `tools/youtube`, `utils`
  - TypeScript: `llm-harness-js`, `src`, `src/agents`, `src/agents/youtube`, `src/clients`, `src/tools`, `src/tools/fs`, `src/tools/web`, `src/tools/youtube`, `src/utils`
- Structural signals (from module stats):
  - TS/JS explicit exports: 36
  - TS/JS import edges: 51 (relative: 26)
  - Python import edges: 160
