"""OpenRouter LLM client initialization and configuration."""

import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter as NativeChatOpenRouter

load_dotenv()

INVALID_MODEL_FORMAT_MESSAGE = "Invalid OpenRouter model format: {model}. Expected PROVIDER/MODEL"


def _is_openrouter(model: str) -> bool:
    """Check if model is OpenRouter format (PROVIDER/MODEL)."""
    return "/" in model and len(model.split("/")) == 2


def _validate_openrouter_model(model: str) -> None:
    if not _is_openrouter(model):
        raise ValueError(INVALID_MODEL_FORMAT_MESSAGE.format(model=model))


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
    """Initialize an OpenRouter chat model using native langchain-openrouter.

    Args:
        model: PROVIDER/MODEL format
        temperature: Sampling temperature (0.0-2.0)
        reasoning_effort: "minimal", "low", "medium", or "high"
        provider_sort: OpenRouter routing - "throughput", "price", "latency"
        web_search: Enable web search plugin
        web_search_engine: "native" (provider built-in), "exa" (Exa API), or None (auto-detect)
        web_search_max_results: Maximum number of search results (default: 5)
        pdf_engine: "mistral-ocr" (scanned), "pdf-text" (structured), "native"
        **kwargs: Additional arguments passed to langchain_openrouter.ChatOpenRouter
    """
    _validate_openrouter_model(model)

    api_key = os.getenv("OPENROUTER_API_KEY")
    explicit_plugins = kwargs.pop("plugins", None)
    openrouter_provider = kwargs.pop("openrouter_provider", None) or {}
    kwargs.pop("model", None)
    kwargs.pop("temperature", None)

    if provider_sort and "sort" not in openrouter_provider:
        openrouter_provider["sort"] = provider_sort

    derived_plugins: list[dict[str, Any]] = []

    if pdf_engine:
        derived_plugins.append({"id": "file-parser", "pdf": {"engine": pdf_engine}})

    if web_search:
        web_plugin: dict = {"id": "web"}
        if web_search_engine:
            web_plugin["engine"] = web_search_engine
        if web_search_max_results != 5:
            web_plugin["max_results"] = web_search_max_results
        derived_plugins.append(web_plugin)

    plugins_by_id: dict[str, dict[str, Any]] = {}
    for plugin in derived_plugins:
        plugin_id = plugin.get("id")
        if plugin_id:
            plugins_by_id[plugin_id] = plugin
    if isinstance(explicit_plugins, list):
        for plugin in explicit_plugins:
            plugin_id = plugin.get("id")
            if plugin_id:
                plugins_by_id[plugin_id] = plugin
            else:
                # Keep anonymous plugins without dedupe semantics.
                plugins_by_id[f"__anon_{id(plugin)}"] = plugin
    merged_plugins = list(plugins_by_id.values())

    model_kwargs: dict[str, Any] = {
        "api_key": api_key,
        **kwargs,
    }
    if reasoning_effort:
        model_kwargs["reasoning"] = {"effort": reasoning_effort}
    if openrouter_provider:
        model_kwargs["openrouter_provider"] = openrouter_provider
    if merged_plugins:
        model_kwargs["plugins"] = merged_plugins

    return NativeChatOpenRouter(
        model=model,
        temperature=temperature,
        **model_kwargs,
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
    _validate_openrouter_model(model)

    api_key = os.getenv("OPENROUTER_API_KEY")
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        check_embedding_ctx_length=kwargs.pop("check_embedding_ctx_length", False),
        **kwargs,
    )
