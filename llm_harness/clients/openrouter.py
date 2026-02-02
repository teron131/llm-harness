"""OpenRouter LLM client initialization and configuration."""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def _is_openrouter(model: str) -> bool:
    """Check if model is OpenRouter format (PROVIDER/MODEL)."""
    return "/" in model and len(model.split("/")) == 2


def _is_gemini(model: str) -> bool:
    """Check if model is a Gemini model."""
    return model.lower().startswith("gemini")


def _get_config(model: str, api_key: str | None = None) -> tuple[str, str]:
    """Get API key and base URL based on model type."""
    if _is_openrouter(model):
        return api_key or os.getenv("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1"

    # Default to Gemini/Google (via OpenAI compatibility endpoint)
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key, "https://generativelanguage.googleapis.com/v1beta/openai/"


def ChatOpenRouter(
    model: str,
    temperature: float = 0,
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None,
    provider_sort: Literal["throughput", "price", "latency"] = "throughput",
    web_search: bool = False,
    web_search_engine: Literal["native", "exa"] | None = None,
    web_search_max_results: int = 5,
    pdf_engine: Literal["mistral-ocr", "pdf-text", "native"] | None = None,
    cached_content: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """Initialize OpenRouter or Gemini model.

    Args:
        model: PROVIDER/MODEL for OpenRouter, or gemini-* for Gemini
        temperature: Sampling temperature (0.0-2.0)
        reasoning_effort: "minimal", "low", "medium", "high"
        provider_sort: OpenRouter routing - "throughput", "price", "latency"
        web_search: Enable web search plugin
        web_search_engine: "native" (provider built-in), "exa" (Exa API), or None (auto-detect)
        web_search_max_results: Maximum number of search results (default: 5)
        pdf_engine: "mistral-ocr" (scanned), "pdf-text" (structured), "native"
        cached_content: Gemini cached content ID
        **kwargs: Additional arguments passed to ChatOpenAI
    """
    if not (_is_openrouter(model) or _is_gemini(model)):
        raise ValueError(f"Invalid model: {model}")

    api_key, base_url = _get_config(model)
    extra_body = kwargs.pop("extra_body", {}) or {}

    if _is_openrouter(model):
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

    elif _is_gemini(model) and cached_content:
        extra_body.setdefault("google", {})["cached_content"] = cached_content

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body or None,
        **kwargs,
    )


def EmbeddingsOpenRouter(
    model: str = "google/gemini-embedding-001",
    api_key: str | None = None,
    **kwargs,
) -> OpenAIEmbeddings:
    """Initialize an OpenRouter embedding model with sensible defaults."""
    if not (_is_openrouter(model) or _is_gemini(model)):
        raise ValueError(f"Invalid model: {model}")

    api_key, base_url = _get_config(model, api_key)
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=kwargs.pop("check_embedding_ctx_length", False),
        **kwargs,
    )


# ============================================================================
# Response parsing utilities
# ============================================================================


def _extract_reasoning(content_blocks: list[dict]) -> str | None:
    """Extract reasoning from response content_blocks."""
    if not content_blocks:
        return None

    block = content_blocks[0]
    if reasoning := block.get("reasoning"):
        return reasoning

    if (
        (extras := block.get("extras"))
        and isinstance(extras, dict)
        and (content := extras.get("content"))
        and isinstance(content, list)
        and content
        and isinstance(content[-1], dict)
    ):
        return content[-1].get("text")
    return None


def parse_invoke(
    response: AIMessage,
    include_reasoning: bool = False,
) -> str | tuple[str | None, str]:
    """Parse response to extract answer and optionally reasoning."""
    if isinstance(response.content, str):
        return (None, response.content) if include_reasoning else response.content

    content_blocks = getattr(response, "content_blocks", [])
    if not content_blocks:
        # Fallback if no content blocks but also not string (unlikely for ChatOpenAI)
        return ("", "") if include_reasoning else ""

    answer = content_blocks[-1].get("text", "")
    if include_reasoning:
        reasoning = _extract_reasoning(content_blocks)
        return reasoning, answer
    return answer


def parse_batch(
    responses: list[AIMessage],
    include_reasoning: bool = False,
) -> list[str] | list[tuple[str | None, str]]:
    """Parse batched responses, optionally with reasoning."""
    return [parse_invoke(response, include_reasoning) for response in responses]
