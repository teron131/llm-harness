# Codemap: `llm_harness/stats/image/image_stats`

## Scope

- Final selected image stats payload pipeline.

## Pipeline

- `source_stage.py`
  - fetch Artificial Analysis and Arena payloads
  - build lookup maps keyed by slug or Arena model name
- `match_stage.py`
  - run the image matcher against fetched source rows
  - attach AA and Arena source rows to matcher output
  - append unmatched Arena rows as fallbacks
- `final_stage.py`
  - project matched rows into the public final payload
  - compute averaged scores and percentiles
  - sort and exact-id filter
- `cache.py`
  - list-mode cache read/write only
- `types.py`
  - staged image payload and handoff types
