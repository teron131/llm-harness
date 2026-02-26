# Codemap: `llm-harness-js/src/agents`

## Scope

- Agent orchestration layer that wraps chat models, tools, and YouTube summarization workflows.

## High-signal locations

- `src/agents/agents.ts -> BaseHarnessAgent/ExaAgent/Web*Agent/ImageAnalysisAgent/YouTubeSummarizer*`
- `src/agents/index.ts -> tool registry + exported surface`
- `src/agents/youtube/index.ts -> top-level YouTube entrypoints`

## Key takeaways per location

- `BaseHarnessAgent` is the core path for model setup and response handling.
- `WebSearchAgent`, `WebLoaderAgent`, and `WebSearchLoaderAgent` vary capabilities via constructor options and tool usage.
- `ImageAnalysisAgent` routes media content through `MediaMessage`.
- YouTube summarizer classes delegate to `agents/youtube/*` modules.
- `agents/index.ts` owns default tool registry (`webloaderTool`, `youtubeloaderTool`).

## Project-specific conventions and rationale

- Keep agent constructors lightweight and pass provider details through `clients/openrouter.ts`.
- Preserve class names aligned with Python equivalents (`YouTubeSummarizer`, `YouTubeSummarizerReAct`, `YouTubeSummarizerGemini`), with aliases acceptable.
- Keep orchestration logic in `agents`, not in `utils` or `tools`.

## Syntax relationship highlights (ast-grep-first)

- `agents.ts -> import ChatOpenRouter from clients/openrouter.ts`
- `agents.ts -> import MediaMessage from clients/multimodal.ts`
- `agents.ts -> import summarizeVideo* from agents/youtube/*`
- `index.ts -> getTools() returns webloaderTool + youtubeloaderTool`

## General approach (not rigid checklist)

- Add new agent variants by extending the base behavior first.
- Keep invoke contracts explicit (`input -> Promise<output>`).
- Prefer delegating multi-step workflows to specialized submodules (`agents/youtube`).

## Validation commands

- `cd llm-harness-js && npm run build`
- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
