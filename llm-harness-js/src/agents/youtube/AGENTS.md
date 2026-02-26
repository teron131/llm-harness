# Codemap: `llm-harness-js/src/agents/youtube`

## Scope

- YouTube summarization flows: lite, ReAct/graph, and Gemini multimodal.

## High-signal locations

- `src/agents/youtube/index.ts -> summarizeVideo/streamSummarizeVideo/summarizeVideoReact/youtubeLoader`
- `src/agents/youtube/summarizerLite.ts -> lightweight summarize flow`
- `src/agents/youtube/summarizer.ts -> graph-based summarize + quality loop`
- `src/agents/youtube/summarizerGemini.ts -> direct Gemini multimodal summarize`
- `src/agents/youtube/schemas.ts -> Summary/Quality/Garbage schemas + scoring helpers`
- `src/agents/youtube/prompts.ts -> prompt templates`

## Key takeaways per location

- `index.ts` is the API boundary expected by higher-level agents.
- `summarizerLite.ts` performs transcript fetch/filter + structured summary generation.
- `summarizer.ts` runs iterative quality validation with `StateGraph`.
- `summarizerGemini.ts` handles native video URL multimodal calls and usage tracking.
- `schemas.ts` is the contract layer for output quality and formatting.

## Project-specific conventions and rationale

- Keep sponsor/garbage filtering as a first-class step.
- Preserve model defaults and quality threshold (`>= 80`) unless intentionally changed.
- Keep prompt text centralized in `prompts.ts`.
- Maintain `Summary` schema compatibility across all summarizer implementations.

## Syntax relationship highlights (ast-grep-first)

- `summarizerLite.ts -> ChatOpenRouter + getTranscript + fastCopy tag/filter/untag`
- `summarizer.ts -> StateGraph + garbage_filter -> summary -> quality`
- `summarizerGemini.ts -> GoogleGenAI + trackUsage`
- `schemas.ts -> TagRangeSchema + s2hk conversion`
- `index.ts -> delegates to summarizer modules`

## General approach (not rigid checklist)

- Prefer schema/prompt updates before rewriting orchestration logic.
- Keep each summarizer focused on one execution style.
- When debugging output quality, inspect: prompt -> schema -> post-filter path.

## Validation commands

- `cd llm-harness-js && npm run build`
- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
