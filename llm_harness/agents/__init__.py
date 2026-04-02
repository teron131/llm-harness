"""Agent package exports and tool registry."""

from langchain.tools import BaseTool, tool

from ..tools.web import webloader, webloader_tool
from .youtube import youtube_loader


@tool(parse_docstring=True)
def youtube_loader_tool(url: str) -> str:
    """Load YouTube transcript and metadata from a video URL.

    Args:
        url: YouTube video URL to load.
    """
    return youtube_loader(url)


def get_tools() -> list[BaseTool]:
    """Return the default tool set used by harness agents."""
    return [webloader_tool, youtube_loader_tool]


__all__ = [
    "get_tools",
    "webloader",
    "webloader_tool",
    "youtube_loader",
    "youtube_loader_tool",
]
