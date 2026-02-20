# Codemap: `llm_harness/agents/youtube`

## Scope

- YouTube summarization pipelines: LangGraph loop, ReAct-lite flow, and Gemini-native multimodal flow.

## High-signal locations

- `llm_harness/agents/youtube/summarizer.py -> create_graph/summarize_video`
- `llm_harness/agents/youtube/summarizer_lite.py -> create_summarizer_agent/summarize_video`
- `llm_harness/agents/youtube/summarizer_gemini.py -> analyze_video_url/summarize_video`
- `llm_harness/agents/youtube/schemas.py -> Summary/Quality/GarbageIdentification`
- `llm_harness/agents/youtube/prompts.py -> prompt builders`
- `llm_harness/agents/youtube/__init__.py -> public entrypoints`

## Key takeaways per location

- `summarizer.py` is the strongest control loop:
  garbage filter -> summary -> quality check -> conditional refinement until acceptable or max iterations.
- `summarizer_lite.py` pushes transcript filtering into tool middleware before final structured summary generation.
- `summarizer_gemini.py` uses Gemini file/video input directly and records token/cost usage via `clients.usage.track_usage`.
- `schemas.py` is the contract layer for all pipelines; quality scoring and acceptance threshold logic lives here.
- `prompts.py` centralizes style and safety instructions, including anti-meta-language and sponsor removal constraints.

## Project-specific conventions and rationale

- Summary output must remain schema-grounded (`Summary` + chronological chapters).
- Garbage/sponsor removal is a first-class invariant, not optional post-processing.
- Quality acceptance is score-based (`Quality.is_acceptable`), with iterative refinement in graph mode.
- Language handling is explicit (`target_language`), including conversion hooks in schema validators.

## Syntax relationship highlights (ast-grep-first)

- `summarizer.py -> garbage_filter_node -> tag_content/filter_content/untag_content`
- `summarizer.py -> summary_node/quality_node -> ChatOpenRouter(...).with_structured_output(...)`
- `summarizer.py -> should_continue -> END or "summary"`
- `summarizer_lite.py -> garbage_filter_middleware -> scrape_youtube tool result mutation`
- `summarizer_gemini.py -> analyze_video_url -> get_gemini_summary_prompt + track_usage`
- `__init__.py -> summarize_video/stream_summarize_video -> summarizer module lazy imports`

## General approach (not rigid checklist)

- Choose `summarizer.py` when you need iterative quality enforcement.
- Choose `summarizer_lite.py` for simpler agentic flows with middleware-based cleaning.
- Choose `summarizer_gemini.py` when visual cues in video matter and Gemini-native multimodal input is preferred.
- Update `prompts.py`/`schemas.py` before altering orchestration logic when output quality drifts.

## Validation commands

- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
- `uv run ruff check .`
