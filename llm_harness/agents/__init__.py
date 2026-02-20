"""Agent package exports and tool registry."""

from langchain_core.tools import BaseTool, tool

from ..tools.web import webloader, webloader_tool
from .youtube import youtube_loader


@tool
def youtubeloader_tool(url: str) -> str:
    """Load YouTube transcript and metadata from a video URL."""
    return youtube_loader(url)


def get_tools() -> list[BaseTool]:
    """Return the default tool set used by harness agents."""
    return [webloader_tool, youtubeloader_tool]


__all__ = [
    "get_tools",
    "webloader",
    "webloader_tool",
    "youtube_loader",
    "youtubeloader_tool",
]
