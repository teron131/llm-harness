# Codemap: `llm_harness/agents`

## Scope

- Agent orchestration layer: wraps model clients and tools into task-level interfaces.

## High-signal locations

- `llm_harness/agents/agents.py -> ExaAgent`
- `llm_harness/agents/agents.py -> BaseHarnessAgent`
- `llm_harness/agents/agents.py -> WebSearchAgent/WebLoaderAgent/WebSearchLoaderAgent`
- `llm_harness/agents/agents.py -> ImageAnalysisAgent`
- `llm_harness/agents/agents.py -> YouTubeSummarizer*`
- `llm_harness/agents/__init__.py -> get_tools/youtubeloader_tool`

## Key takeaways per location

- `BaseHarnessAgent` is the core constructor path: chooses model, sets optional tool list, and handles structured vs plain response extraction.
- Web agents differ mainly by enabled capabilities (`web_search`, `webloader_tool`) while sharing invocation shape.
- YouTube summarizer agents delegate to `agents/youtube/*` implementations rather than embedding pipeline logic in `agents.py`.
- `agents/__init__.py` provides a lightweight tool registry and wrappers for LangChain tool usage.

## Project-specific conventions and rationale

- Keep agent constructors simple and compositional; avoid embedding provider-specific mechanics here.
- Preserve model names/constants unless explicitly requested.
- Prefer delegating heavy flow logic to submodules (`agents/youtube`) to keep top-level agent classes thin.

## Syntax relationship highlights (ast-grep-first)

- `agents.py -> BaseHarnessAgent.__init__ -> ChatOpenRouter`
- `agents.py -> Web*Agent.invoke -> self.agent.invoke -> _process_response`
- `agents.py -> ImageAnalysisAgent.invoke -> MediaMessage`
- `agents.py -> YouTubeSummarizerReAct.invoke -> youtube.summarizer_react.summarize_video_react`
- `agents/__init__.py -> youtubeloader_tool -> youtube_loader`

## General approach (not rigid checklist)

- For new agent variants, start by extending `BaseHarnessAgent` and pass differences as constructor args.
- Keep invocation contracts explicit (`str` input, `BaseModel | str` output) and avoid ad-hoc response parsing.
- Use `agents/youtube` for multi-step logic, retries, and quality loops.

## Validation commands

- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
- `uv run ruff check .`
