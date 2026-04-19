# llm-harness

Python-first LLM toolkit shaped by accumulated work across multiple projects, not a generic one-size-fits-all harness.

## Structure

```text
llm_harness/
├── agents/    workflow orchestration
│   ├── fixer/
│   └── youtube/
├── clients/   provider adapters
├── stats/     LLM and image stats
│   ├── llm/
│   └── image/
├── tools/     integration boundaries
│   ├── fs/
│   ├── sql/
│   ├── tabular/
│   ├── web/
│   └── youtube/
└── utils/     deterministic helpers
```

## Install

```bash
uv sync
```

## Editable use from another repo

```bash
uv add --editable /path/to/llm-harness
uv sync
```

## Verification

```bash
uv run ruff check .
uv run ruff format .
```
