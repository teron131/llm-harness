"""YouTube video transcript summarization using LangChain ReAct agent with structured output."""

from collections.abc import Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ...clients import ChatOpenRouter
from ...fast_copy import (
    filter_content,
    tag_content,
    untag_content,
)
from .prompts import get_garbage_filter_prompt, get_langchain_summary_prompt
from .schemas import GarbageIdentification, Summary
from .scrapper import scrape_youtube as _scrape_youtube
from .utils import is_youtube_url

load_dotenv()

MODEL = "google/gemini-3-flash-preview"
FAST_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"


@tool
def scrape_youtube(youtube_url: str) -> str:
    """Scrape a YouTube video and return the transcript.

    Args:
        youtube_url: The YouTube video URL to scrape

    Returns:
        Parsed transcript text
    """
    result = _scrape_youtube(youtube_url)
    if not result.has_transcript:
        raise ValueError("Video has no transcript")
    if not result.parsed_transcript:
        raise ValueError("Transcript is empty")
    return result.parsed_transcript


@wrap_tool_call
def garbage_filter_middleware(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """Middleware to filter garbage from tool results (like transcripts)."""
    result = handler(request)

    # Only filter if it's the scrape_youtube tool and the call succeeded
    if request.tool_call["name"] == "scrape_youtube" and result.status != "error":
        transcript = result.content
        if isinstance(transcript, str) and transcript.strip():
            # Apply the tagging/filtering mechanism
            tagged_transcript = tag_content(transcript)

            llm = ChatOpenRouter(
                model=FAST_MODEL,
                temperature=0,
            ).with_structured_output(GarbageIdentification)

            system_prompt = get_garbage_filter_prompt()

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=tagged_transcript),
            ]

            garbage: GarbageIdentification = llm.invoke(messages)

            if garbage.garbage_ranges:
                filtered_transcript = filter_content(tagged_transcript, garbage.garbage_ranges)
                cleaned_transcript = untag_content(filtered_transcript)
                print(f"🧹 Middleware removed {len(garbage.garbage_ranges)} garbage sections from tool result.")
                # Update the result content
                result.content = cleaned_transcript

    return result


def create_summarizer_agent(target_language: str | None = None):
    """Create a ReAct agent for summarizing video transcripts with structured output."""
    llm = ChatOpenRouter(
        model=MODEL,
        temperature=0,
        reasoning_effort="medium",
    )

    system_prompt = get_langchain_summary_prompt(target_language=target_language)

    agent = create_agent(
        model=llm,
        tools=[scrape_youtube],
        system_prompt=system_prompt,
        middleware=[garbage_filter_middleware],  # Add the garbage filter middleware
        response_format=ToolStrategy(Summary),  # Use ToolStrategy for better error handling
    )

    return agent


def _parse_result(summary: Summary) -> str:
    """Format Summary object into a readable string.

    Args:
        summary: The Summary object to format

    Returns:
        Formatted string representation of the summary
    """
    lines = [
        "=" * 80,
        "SUMMARY:",
        "=" * 80,
        f"\nOverview:\n{summary.overview}",
        f"\nChapters ({len(summary.chapters)}):",
    ]

    for i, chapter in enumerate(summary.chapters, 1):
        lines.append(f"\n  Chapter {i}: {chapter.title}")
        lines.append(f"    Summary: {chapter.description}")
        if chapter.start_time or chapter.end_time:
            time_range = f"{chapter.start_time or '?'} - {chapter.end_time or '?'}"
            lines.append(f"    Time: {time_range}")

    return "\n".join(lines)


def summarize_video(
    transcript_or_url: str,
    target_language: str | None = None,
) -> str:
    """Summarize video transcript or YouTube URL.

    Args:
        transcript_or_url: Transcript text or YouTube URL
        target_language: Optional target language code (e.g., "en", "es", "fr")

    Returns:
        Formatted string representation of the summary
    """
    agent = create_summarizer_agent(target_language)

    # If it's a YouTube URL, let the agent use the tool to fetch it
    # Otherwise, provide the transcript directly
    prompt = f"Summarize this YouTube video: {transcript_or_url}" if is_youtube_url(transcript_or_url) else f"Transcript:\n{transcript_or_url}"

    response = agent.invoke({"messages": [HumanMessage(content=prompt)]})

    # Extract structured response
    structured_response = response.get("structured_response")
    if structured_response is None:
        raise ValueError("Agent did not return structured response")

    return _parse_result(structured_response)


if __name__ == "__main__":
    # Example usage
    video_url = "https://youtu.be/UALxgn1MnZo"
    print(f"Summarizing: {video_url}\n")

    result = summarize_video(video_url)
    print(result)
