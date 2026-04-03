"""Agent package exports and tool registry."""

from langchain.tools import BaseTool, tool

from ..tools.web import webloader, webloader_tool
from .tabular_agent import TabularTaskAgent, TabularTaskInput, TabularTaskOutput, build_task_prompt, run_task
from .youtube import youtubeloader


@tool(parse_docstring=True)
def youtubeloader_tool(url: str) -> str:
    """Load YouTube transcript and metadata from a video URL.

    Args:
        url: YouTube video URL to load.
    """
    return youtubeloader(url)


def get_tools() -> list[BaseTool]:
    """Return the default tool set used by harness agents."""
    return [webloader_tool, youtubeloader_tool]


__all__ = [
    "TabularTaskAgent",
    "TabularTaskInput",
    "TabularTaskOutput",
    "build_task_prompt",
    "get_tools",
    "run_task",
    "webloader",
    "webloader_tool",
    "youtubeloader",
    "youtubeloader_tool",
]
