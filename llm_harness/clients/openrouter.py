"""OpenRouter LLM client initialization and configuration."""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def _is_openrouter(model: str) -> bool:
    """Check if model is OpenRouter format (PROVIDER/MODEL)."""
    return "/" in model and len(model.split("/")) == 2


def ChatOpenRouter(
    model: str,
    temperature: float = 0.7,
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None,
    provider_sort: Literal["throughput", "price", "latency"] = "throughput",
    web_search: bool = False,
    web_search_engine: Literal["native", "exa"] | None = None,
    web_search_max_results: int = 5,
    pdf_engine: Literal["mistral-ocr", "pdf-text", "native"] | None = None,
    **kwargs,
) -> BaseChatModel:
    """Initialize OpenRouter model.

    Args:
        model: PROVIDER/MODEL format
        temperature: Sampling temperature (0.0-2.0)
        reasoning_effort: "minimal", "low", "medium", "high"
        provider_sort: OpenRouter routing - "throughput", "price", "latency"
        web_search: Enable web search plugin
        web_search_engine: "native" (provider built-in), "exa" (Exa API), or None (auto-detect)
        web_search_max_results: Maximum number of search results (default: 5)
        pdf_engine: "mistral-ocr" (scanned), "pdf-text" (structured), "native"
        **kwargs: Additional arguments passed to ChatOpenAI
    """
    if not _is_openrouter(model):
        raise ValueError(f"Invalid OpenRouter model format: {model}. Expected PROVIDER/MODEL")

    api_key = os.getenv("OPENROUTER_API_KEY")
    extra_body = kwargs.pop("extra_body", {}) or {}

    if provider_sort and "provider" not in extra_body:
        extra_body["provider"] = {"sort": provider_sort}

    plugins = extra_body.get("plugins", [])

    if pdf_engine:
        plugins.append({"id": "file-parser", "pdf": {"engine": pdf_engine}})

    if web_search:
        web_plugin: dict = {"id": "web"}
        if web_search_engine:
            web_plugin["engine"] = web_search_engine
        if web_search_max_results != 5:
            web_plugin["max_results"] = web_search_max_results
        plugins.append(web_plugin)

    if plugins:
        extra_body["plugins"] = plugins

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body or None,
        **kwargs,
    )


def OpenRouterEmbeddings(
    model: str = "openai/text-embedding-3-small",
    **kwargs,
) -> OpenAIEmbeddings:
    """Initialize an OpenRouter embedding model.

    Args:
        model: OpenRouter embedding model in PROVIDER/MODEL format
        **kwargs: Additional arguments passed to OpenAIEmbeddings
    """
    if not _is_openrouter(model):
        raise ValueError(f"Invalid OpenRouter model format: {model}. Expected PROVIDER/MODEL")

    api_key = os.getenv("OPENROUTER_API_KEY")
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        check_embedding_ctx_length=kwargs.pop("check_embedding_ctx_length", False),
        **kwargs,
    )
