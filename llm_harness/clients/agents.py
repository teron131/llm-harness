"""Pre-configured agents for common LLM tasks."""

import os
from pathlib import Path
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from ..tools.web import webloader_tool
from . import ChatOpenRouter, MediaMessage

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class BaseHarnessAgent:
    """Base class for harness agents."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0,
        reasoning_effort: ReasoningEffort = "medium",
        system_prompt: str | None = None,
        response_format: type[BaseModel] | None = None,
        tools: list[BaseTool] | None = None,
        **model_kwargs: Any,
    ):
        model = model or os.getenv("FAST_LLM")
        self.system_prompt = system_prompt
        self.response_format = response_format
        self.model = ChatOpenRouter(
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            **model_kwargs,
        )
        self.agent = create_agent(
            model=self.model,
            tools=tools or [],
            system_prompt=self.system_prompt,
            response_format=ToolStrategy(self.response_format) if self.response_format else None,
        )

    def _process_response(self, response: dict) -> BaseModel | str:
        if self.response_format:
            return response.get("structured_response")
        return response.get("messages")[-1].content


class WebSearchAgent(BaseHarnessAgent):
    """Agent that uses web search capabilities to find and process information."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0,
        reasoning_effort: ReasoningEffort = "medium",
        system_prompt: str | None = None,
        response_format: type[BaseModel] | None = None,
        web_search_max_results: int = 5,
        **model_kwargs: Any,
    ):
        """Initialize web search agent."""
        super().__init__(
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            system_prompt=system_prompt,
            response_format=response_format,
            web_search=True,
            web_search_max_results=web_search_max_results,
            **model_kwargs,
        )

    def invoke(self, user_input: str) -> BaseModel | str:
        """Execute web search and process results."""
        response = self.agent.invoke({"messages": [HumanMessage(content=user_input)]})
        return self._process_response(response)


class WebLoaderAgent(BaseHarnessAgent):
    """Agent that loads web content from URLs and processes it using tools."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0,
        reasoning_effort: ReasoningEffort = "medium",
        system_prompt: str | None = None,
        response_format: type[BaseModel] | None = None,
        **model_kwargs: Any,
    ):
        """Initialize web loader agent with tool capabilities."""

        super().__init__(
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            system_prompt=system_prompt,
            response_format=response_format,
            tools=[webloader_tool],
            **model_kwargs,
        )

    def invoke(self, user_input: str) -> BaseModel | str:
        """Load and process web content based on user input."""
        response = self.agent.invoke({"messages": [HumanMessage(content=user_input)]})
        return self._process_response(response)


class ImageAnalysisAgent(BaseHarnessAgent):
    """Agent that accepts image inputs and returns structured or plain responses."""

    def invoke(self, image_paths: str | Path | list[str | Path], description: str = "") -> BaseModel | str:
        """Analyze one or more images with an optional description/prompt."""
        response = self.agent.invoke({"messages": [MediaMessage(paths=image_paths, description=description)]})
        return self._process_response(response)
