"""YouTube video summarization using Google Gemini multimodal API.

This module provides a direct interface to Gemini's native video analysis
capabilities, allowing you to pass YouTube URLs directly to the model.
"""

import logging
import os
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3-flash-preview"

# Pricing in USD per million tokens
USD_PER_M_TOKENS_BY_MODEL = {
    "gemini-3-flash-preview": {"input": 0.5, "output": 3},
    "gemini-3-pro-preview": {"input": 2, "output": 12},
}


class Chapter(BaseModel):
    """Represents a single chapter in the video summary."""

    title: str = Field(
        description="A concise chapter heading.",
    )
    description: str = Field(
        description="A substantive chapter description grounded in the content. Include key facts (numbers/names/steps) when present. Avoid meta-language like 'the video', 'the author', 'the speaker says'—state the content directly.",
    )
    start_time: str | None = Field(
        None,
        description="Optional chapter start timestamp in the format MM:SS.",
    )
    end_time: str | None = Field(
        None,
        description="Optional chapter end timestamp matching the same format as start_time.",
    )


class VideoAnalysis(BaseModel):
    """Complete analysis of a YouTube video."""

    overview: str = Field(
        description="An end-to-end summary of the whole content (main thesis + arc), written in direct statements without meta-language.",
    )
    chapters: list[Chapter] = Field(
        min_length=1,
        description="Chronological, non-overlapping chapters covering the core content.",
    )


def _build_prompt(
    target_language: str = "auto",
    title: str | None = None,
    description: str | None = None,
) -> str:
    """Build the system prompt for video analysis."""
    # Language descriptions mapping
    lang_descriptions = {
        "auto": "Use the same language as the video, or English if the language is unclear",
        "en": "English (US)",
        "zh-TW": "Traditional Chinese (繁體中文)",
    }

    # Determine language instruction
    lang_desc = lang_descriptions.get(target_language, target_language)
    instruction = lang_desc if target_language == "auto" else f"Write ALL output in {lang_desc}. Do not use English or any other language."

    language_instruction = f"- OUTPUT LANGUAGE (REQUIRED): {instruction}"

    metadata_parts = []
    if title:
        metadata_parts.append(f"Video Title: {title}")
    if description:
        metadata_parts.append(f"Video Description: {description}")

    metadata = ""
    if metadata_parts:
        metadata = "\n# CONTEXTUAL INFORMATION:\n" + "\n".join(metadata_parts) + "\n"

    prompt_lines = [
        "Create a grounded, chronological summary.",
        metadata,
        language_instruction,
        "",
        "SOURCE: You are given the full video. Use BOTH spoken content and visuals (on-screen text/slides/charts/code/UI). Do not invent details that are not clearly supported by what you can see/hear.",
        "",
        "Return JSON only (no extra text) with:",
        "- overview: string",
        "- chapters: array of { title: string, description: string, start_time?: string, end_time?: string }",
        "(start_time/end_time are optional MM:SS; omit if unsure)",
        "",
        "Rules:",
        "- Chapters must be chronological and non-overlapping",
        "- Avoid meta-language (no 'this video...' framing)",
        "- Exclude sponsors/promos/calls to action entirely",
    ]

    return "\n".join(prompt_lines)


def _calculate_cost(
    model: str,
    prompt_tokens: int,
    total_tokens: int,
) -> dict | None:
    """Calculate estimated cost based on usage."""
    pricing = USD_PER_M_TOKENS_BY_MODEL.get(model)
    if not pricing:
        return None

    output_tokens = max(0, total_tokens - prompt_tokens)
    estimated_usd = (prompt_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]

    return {
        "currency": "USD",
        "model": model,
        "pricing_usd_per_m_tokens": pricing,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "estimated_usd": estimated_usd,
    }


def analyze_video_url(
    video_url: str,
    *,
    model: str = DEFAULT_MODEL,
    thinking_level: Literal["minimal", "low", "medium", "high"] = "medium",
    target_language: str = "auto",
    video_title: str | None = None,
    video_description: str | None = None,
    api_key: str | None = None,
    timeout: int = 600,
) -> VideoAnalysis | None:
    """Analyze a YouTube video using Gemini's multimodal API.

    Args:
        video_url: YouTube URL (e.g., "https://youtu.be/MiUHjLxm3V0")
        model: Gemini model to use
        thinking_level: Level of reasoning ("minimal", "low", "medium", "high")
        target_language: Target output language ("auto", "en", "zh-TW", or other language code)
        video_title: Optional video title for additional context
        video_description: Optional video description for additional context
        api_key: Gemini API key (defaults to GOOGLE_API_KEY or GEMINI_API_KEY env var)
        timeout: Request timeout in seconds

    Returns:
        VideoAnalysis object or None if analysis fails
    """
    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY")

    client = genai.Client(api_key=api_key, http_options={"timeout": timeout})

    system_prompt = _build_prompt(
        target_language=target_language,
        title=video_title,
        description=video_description,
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part(file_data=types.FileData(file_uri=video_url)),
                types.Part(text=system_prompt),
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                response_mime_type="application/json",
                response_schema=VideoAnalysis,
            ),
        )

        if not response.text:
            logger.warning("Empty response from Gemini API")
            return None

        # Parse the response into our Pydantic model
        analysis = VideoAnalysis.model_validate_json(response.text)

        # Log usage and cost information
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            logger.info(f"Usage metadata: {usage}")

            if hasattr(usage, "prompt_token_count") and hasattr(usage, "total_token_count"):
                cost_info = _calculate_cost(
                    model,
                    usage.prompt_token_count,
                    usage.total_token_count,
                )
                if cost_info:
                    logger.info(f"Estimated cost: ${cost_info['estimated_usd']:.4f} USD")
                    logger.info(f"Tokens - Input: {cost_info['prompt_tokens']}, Output: {cost_info['output_tokens']}")

        return analysis

    except Exception as e:
        logger.exception(f"Failed to analyze video: {e}")
        return None


def summarize_youtube_video(
    video_url: str,
    *,
    model: str = DEFAULT_MODEL,
    thinking_level: Literal["minimal", "low", "medium", "high"] = "medium",
    target_language: str = "auto",
    video_title: str | None = None,
    video_description: str | None = None,
    api_key: str | None = None,
) -> VideoAnalysis | None:
    """Convenience function to summarize a YouTube video.

    Args:
        video_url: YouTube URL
        model: Gemini model to use
        thinking_level: Reasoning level
        target_language: Target output language ("auto", "en", "zh-TW")
        video_title: Optional video title for context
        video_description: Optional video description for context
        api_key: Optional API key override

    Returns:
        VideoAnalysis with overview and chapters
    """
    return analyze_video_url(
        video_url,
        model=model,
        thinking_level=thinking_level,
        target_language=target_language,
        video_title=video_title,
        video_description=video_description,
        api_key=api_key,
    )
