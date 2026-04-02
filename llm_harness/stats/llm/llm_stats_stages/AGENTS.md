# Codemap: `llm_harness/stats/llm/llm_stats_stages`

## Scope

- Native Python staged LLM stats pipeline.

## Pipeline

- `source_stage.py`
  - fetch native Artificial Analysis and models.dev payloads
- `match_stage.py`
  - build the native matched/union LLM rows
- `openrouter_stage.py`
  - hold late OpenRouter enrichment hooks for the LLM pipeline
- `final_stage.py`
  - filter/project the final selected payload
- `types.py`
  - shared stage types/configs
