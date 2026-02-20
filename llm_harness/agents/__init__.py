from functools import wraps

from langchain_core.tools import BaseTool, tool
from rich import print

from ..tools.web import webloader, webloader_tool
from .youtube import youtube_loader


def print_tool_info(tool_func: BaseTool) -> None:
    print(f"Tool:{tool_func.name}")
    print(f"Description: {tool_func.description}")
    print(f"Arguments: {tool_func.args}\n")


@tool
@wraps(youtube_loader)
def youtubeloader_tool(url: str) -> str:
    return youtube_loader(url)


def get_tools() -> list[BaseTool]:
    """Get the list of available tools for the UniversalChain. The tools are wrapped with their original docstrings and registered as langchain tools.

    Returns:
        List[BaseTool]: List of tool functions with their docstrings preserved
    """

    tools = [webloader_tool, youtubeloader_tool]

    # Print info for all tools
    # for tool_func in tools:
    #     print_tool_info(tool_func)

    return tools


__all__ = [
    "get_tools",
    "webloader",
    "webloader_tool",
    "youtube_loader",
    "youtubeloader_tool",
]
