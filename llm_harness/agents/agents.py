"""Pre-configured agents for common LLM tasks."""

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from exa_py import Exa
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from ..clients.multimodal import MediaMessage
from ..clients.openrouter import ChatOpenRouter
from ..tools.web import webloader_tool
from .youtube.schemas import Summary
from .youtube.summarizer import summarize_video as summarize_video_react
from .youtube.summarizer_gemini import summarize_video as summarize_video_gemini
from .youtube.summarizer_lite import summarize_video as summarize_video_lite

load_dotenv()

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class ExaAgent:
    """Exa API as web search subagent."""

    def __init__(self, system_prompt: str, output_schema: type[BaseModel]):
        self.exa = Exa(api_key=os.getenv("EXA_API_KEY"))
        self.system_prompt = system_prompt
        self.output_schema = output_schema

    def invoke(self, query: str) -> BaseModel:
        result = self.exa.answer(
            query=query,
            system_prompt=self.system_prompt,
            output_schema=self.output_schema.model_json_schema(),
        )
        return self.output_schema.model_validate(result.answer)


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
        if not model:
            raise ValueError("No model configured. Pass `model=...` or set `FAST_LLM`.")
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
        web_search_engine: Literal["native", "exa"] | None = None,
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
            web_search_engine=web_search_engine,
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


class WebSearchLoaderAgent(BaseHarnessAgent):
    """Agent with both web search and web loader tool enabled."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0,
        reasoning_effort: ReasoningEffort = "medium",
        system_prompt: str | None = None,
        response_format: type[BaseModel] | None = None,
        web_search_engine: Literal["native", "exa"] | None = None,
        web_search_max_results: int = 5,
        **model_kwargs: Any,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            system_prompt=system_prompt,
            response_format=response_format,
            tools=[webloader_tool],
            web_search=True,
            web_search_engine=web_search_engine,
            web_search_max_results=web_search_max_results,
            **model_kwargs,
        )

    def invoke(self, user_input: str) -> BaseModel | str:
        """Run with both web-search and web-loader capabilities."""
        response = self.agent.invoke({"messages": [HumanMessage(content=user_input)]})
        return self._process_response(response)


class ImageAnalysisAgent(BaseHarnessAgent):
    """Agent that accepts image inputs and returns structured or plain responses."""

    def invoke(
        self,
        image_paths: str | Path | list[str | Path],
        description: str = "",
    ) -> BaseModel | str:
        """Analyze one or more images with an optional description/prompt."""
        response = self.agent.invoke({"messages": [MediaMessage(paths=image_paths, description=description)]})
        return self._process_response(response)


class YouTubeSummarizerReActAgent:
    """ReAct-based YouTube summarizer using the LangGraph workflow."""

    def __init__(self, target_language: str | None = None):
        self.target_language = target_language

    def invoke(self, transcript_or_url: str) -> Summary:
        """Summarize a transcript or YouTube URL."""
        return summarize_video_react(
            transcript_or_url=transcript_or_url,
            target_language=self.target_language,
        )


class YouTubeSummarizerLiteAgent:
    """Lightweight ReAct-based YouTube summarizer."""

    def __init__(self, target_language: str | None = None):
        self.target_language = target_language

    def invoke(self, transcript_or_url: str) -> str:
        """Summarize a transcript or YouTube URL."""
        return summarize_video_lite(
            transcript_or_url=transcript_or_url,
            target_language=self.target_language,
        )


class YouTubeSummarizerGeminiAgent:
    """Gemini multimodal YouTube summarizer."""

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        thinking_level: ReasoningEffort = "medium",
        target_language: str = "auto",
        api_key: str | None = None,
    ):
        self.model = model
        self.thinking_level = thinking_level
        self.target_language = target_language
        self.api_key = api_key

    def invoke(self, video_url: str) -> Summary | None:
        """Summarize a YouTube URL using Gemini multimodal input."""
        return summarize_video_gemini(
            video_url=video_url,
            model=self.model,
            thinking_level=self.thinking_level,
            target_language=self.target_language,
            api_key=self.api_key,
        )
