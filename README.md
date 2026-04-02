# llm-harness

Python-only LLM harness package with agents, provider clients, tools, and stats helpers.

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

The TypeScript port now lives in the sibling repo `../llm-harness-js`.
