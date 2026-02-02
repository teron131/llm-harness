"""YouTube video transcript summarization using LangChain with LangGraph self-checking workflow."""

from collections.abc import Generator
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from ...clients import ChatOpenRouter
from ...fast_copy import TagRange, filter_content, tag_content, untag_content
from ...text_utils import s2hk
from .scrapper import YouTubeScrapperResult, scrape_youtube
from .utils import is_youtube_url

# ============================================================================
# Configuration
# ============================================================================

SUMMARY_MODEL = "x-ai/grok-4.1-fast"
QUALITY_MODEL = "x-ai/grok-4.1-fast"
FAST_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"
MIN_QUALITY_SCORE = 80
MAX_ITERATIONS = 2
TARGET_LANGUAGE = "en"  # ISO language code (en, es, fr, de, etc.)


# ============================================================================
# Data Models
# ============================================================================


class Chapter(BaseModel):
    """Represents a single chapter in the summary."""

    header: str = Field(description="A concise chapter heading.")
    summary: str = Field(
        description="A substantive chapter description grounded in the content. Include key facts (numbers/names/steps) when present. Avoid meta-language like 'the video', 'the author', 'the speaker says'—state the content directly."
    )
    key_points: list[str] = Field(description="Important takeaways and insights from this chapter")

    @field_validator("header", "summary")
    def convert_string_to_hk(cls, value: str) -> str:
        """Convert string fields to Traditional Chinese."""
        return s2hk(value)

    @field_validator("key_points")
    def convert_list_to_hk(cls, value: list[str]) -> list[str]:
        """Convert list fields to Traditional Chinese."""
        return [s2hk(item) for item in value]


class Summary(BaseModel):
    """Complete summary of video content."""

    title: str = Field(description="The main title or topic of the video content")
    summary: str = Field(description="An end-to-end summary of the whole content (main thesis + arc), written in direct statements without meta-language.")
    takeaways: list[str] = Field(
        description="Key insights and actionable takeaways for the audience",
        min_length=3,
        max_length=8,
    )
    chapters: list[Chapter] = Field(description="Chronological, non-overlapping chapters covering the core content.")
    keywords: list[str] = Field(
        description="The most relevant keywords in the summary worthy of highlighting",
        min_length=3,
        max_length=3,
    )
    target_language: str | None = Field(default=None, description="The language the content to be translated to")

    @field_validator("title", "summary")
    def convert_string_to_hk(cls, value: str) -> str:
        """Convert string fields to Traditional Chinese."""
        return s2hk(value)

    @field_validator("takeaways", "keywords")
    def convert_list_to_hk(cls, value: list[str]) -> list[str]:
        """Convert list fields to Traditional Chinese."""
        return [s2hk(item) for item in value]


class Rate(BaseModel):
    """Quality rating for a single aspect."""

    rate: Literal["Fail", "Refine", "Pass"] = Field(description="Score for the quality aspect")
    reason: str = Field(description="Reason for the score")


class Quality(BaseModel):
    """Quality assessment of the summary."""

    completeness: Rate = Field(description="Rate for completeness: The entire transcript has been considered")
    structure: Rate = Field(description="Rate for structure: The result is in desired structures")
    no_garbage: Rate = Field(
        description="Rate for no_garbage: The promotional and meaningless content such as cliche intros, outros, filler, sponsorships, and other irrelevant segments are effectively removed"
    )
    meta_language_avoidance: Rate = Field(description="Rate for meta-language avoidance: No phrases like 'This chapter introduces', 'This section covers', etc.")
    useful_keywords: Rate = Field(description="Rate for keywords: The keywords are useful for highlighting the summary")
    correct_language: Rate = Field(description="Rate for language: Match the original language of the transcript or user requested")

    @property
    def all_aspects(self) -> list[Rate]:
        """Return all quality aspects as a list."""
        return [
            self.completeness,
            self.structure,
            self.no_garbage,
            self.meta_language_avoidance,
            self.useful_keywords,
            self.correct_language,
        ]

    @property
    def percentage_score(self) -> int:
        """Calculate percentage score based on Pass/Refine/Fail ratings."""
        aspects = self.all_aspects
        pass_count = sum(1 for a in aspects if a.rate == "Pass")
        refine_count = sum(1 for a in aspects if a.rate == "Refine")
        # Pass = 100%, Refine = 50%, Fail = 0%
        return int((pass_count * 100 + refine_count * 50) / len(aspects))

    @property
    def is_acceptable(self) -> bool:
        """Check if quality score meets minimum threshold."""
        return self.percentage_score >= MIN_QUALITY_SCORE


class GarbageIdentification(BaseModel):
    """List of identified garbage sections in a transcript."""

    garbage_ranges: list[TagRange] = Field(description="List of line ranges identified as promotional or irrelevant content")


class SummarizerState(BaseModel):
    """State schema for the summarization graph."""

    transcript: str | None = None
    summary: Summary | None = None
    quality: Quality | None = None
    target_language: str | None = None
    iteration_count: int = 0
    is_complete: bool = False


class SummarizerOutput(BaseModel):
    """Output schema for the summarization graph."""

    summary: Summary
    quality: Quality | None = None
    iteration_count: int
    transcript: str | None = None


# ============================================================================
# Graph Nodes
# ============================================================================


def garbage_filter_node(state: SummarizerState) -> dict:
    """Identify and remove garbage from the transcript."""
    # Tag the transcript for identification
    tagged_transcript = tag_content(state.transcript)

    llm = ChatOpenRouter(
        model=FAST_MODEL,
        temperature=0,
        reasoning_effort="low",
    ).with_structured_output(GarbageIdentification)

    system_prompt = (
        "Identify transcript lines that are NOT part of the core content and should be removed.\n"
        "Focus on: sponsors/ads/promos, discount codes, affiliate links, subscribe/like/call to action blocks, filler intros/outros, housekeeping, and other irrelevant segments.\n"
        "The transcript contains line tags like [L1], [L2], etc.\n"
        "Return ONLY the line ranges to remove (garbage_ranges).\n"
        "If unsure about a segment, prefer excluding it."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=tagged_transcript),
    ]

    garbage: GarbageIdentification = llm.invoke(messages)

    if garbage.garbage_ranges:
        filtered_transcript = filter_content(tagged_transcript, garbage.garbage_ranges)
        # Untag for the next stage (summary)
        cleaned_transcript = untag_content(filtered_transcript)
        print(f"🧹 Removed {len(garbage.garbage_ranges)} garbage sections.")
        return {"transcript": cleaned_transcript}

    return {}


def summary_node(state: SummarizerState) -> dict:
    """Generate summary from transcript."""
    llm = ChatOpenRouter(
        model=SUMMARY_MODEL,
        temperature=0,
        reasoning_effort="medium",
    ).with_structured_output(Summary)

    system_prompt = (
        "Create a grounded, chronological summary of the transcript.\n"
        "Rules:\n"
        "- Ground every claim in the transcript; do not add unsupported details\n"
        "- Exclude sponsors/ads/promos/calls to action entirely\n"
        "- Avoid meta-language (no 'this video...', 'the speaker...', etc.)\n"
        "- Prefer concrete facts, names, numbers, and steps when present\n"
        "- Ensure output matches the provided response schema"
    )
    if state.target_language:
        system_prompt += f"\nOUTPUT LANGUAGE (REQUIRED): {state.target_language}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Transcript:\n{state.transcript}"),
    ]

    summary = llm.invoke(messages)

    return {
        "summary": summary,
        "iteration_count": state.iteration_count + 1,
    }


def quality_node(state: SummarizerState) -> dict:
    """Assess quality of summary."""
    llm = ChatOpenRouter(
        model=QUALITY_MODEL,
        temperature=0,
        reasoning_effort="low",
    ).with_structured_output(Quality)

    system_prompt = (
        "Evaluate the summary against the transcript.\n"
        "For each aspect in the response schema, return a rating (Fail/Refine/Pass) and a specific, actionable reason.\n"
        "Rules:\n"
        "- Be strict about transcript grounding\n"
        "- Treat any sponsor/promo/call to action content as a failure for no_garbage\n"
        "- Treat meta-language as a failure for meta_language_avoidance"
    )
    if state.target_language:
        system_prompt += f"\nVerify the output language matches: {state.target_language}"

    summary_json = state.summary.model_dump_json() if state.summary else "No summary provided"
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Transcript:\n{state.transcript}\n\nSummary:\n{summary_json}"),
    ]

    quality: Quality = llm.invoke(messages)

    return {
        "quality": quality,
        "is_complete": quality.is_acceptable,
    }


# ============================================================================
# Graph Construction
# ============================================================================


def should_continue(state: SummarizerState) -> str:
    """Determine next step in workflow."""
    quality_percent = state.quality.percentage_score if state.quality else None
    quality_display = f"{quality_percent}%" if quality_percent is not None else "N/A"

    if state.is_complete:
        print(f"✅ Complete: quality {quality_display}")
        return END

    if state.quality and not state.quality.is_acceptable and state.iteration_count < MAX_ITERATIONS:
        print(f"🔄 Refining: quality {quality_display} < {MIN_QUALITY_SCORE}% (iteration {state.iteration_count + 1})")
        return "summary"

    print(f"⚠️ Stopping: quality {quality_display}, {state.iteration_count} iterations")
    return END


def create_graph() -> StateGraph:
    """Create the summarization workflow graph with conditional routing."""
    builder = StateGraph(
        SummarizerState,
        output_schema=SummarizerOutput,
    )

    builder.add_node("garbage_filter", garbage_filter_node)
    builder.add_node("summary", summary_node)
    builder.add_node("quality", quality_node)

    builder.add_edge(START, "garbage_filter")
    builder.add_edge("garbage_filter", "summary")
    builder.add_edge("summary", "quality")

    builder.add_conditional_edges(
        "quality",
        should_continue,
        {
            "summary": "summary",
            END: END,
        },
    )

    return builder.compile()


# ============================================================================
# Helper Functions
# ============================================================================


def _extract_transcript(transcript_or_url: str) -> str:
    """Extract transcript from URL or return text directly."""
    if is_youtube_url(transcript_or_url):
        result: YouTubeScrapperResult = scrape_youtube(transcript_or_url)
        if not result.has_transcript:
            raise ValueError("Video has no transcript")
        if not result.parsed_transcript:
            raise ValueError("Transcript is empty")
        return result.parsed_transcript

    if not transcript_or_url or not transcript_or_url.strip():
        raise ValueError("Transcript cannot be empty")

    return transcript_or_url


# ============================================================================
# Public API
# ============================================================================


def summarize_video(
    transcript_or_url: str,
    target_language: str | None = None,
) -> Summary:
    """Summarize YouTube video or text transcript with quality self-checking."""
    graph = create_graph()
    transcript = _extract_transcript(transcript_or_url)

    initial_state = SummarizerState(
        transcript=transcript,
        target_language=target_language or TARGET_LANGUAGE,
    )
    result: dict = graph.invoke(initial_state.model_dump())
    output = SummarizerOutput.model_validate(result)

    quality_percent = output.quality.percentage_score if output.quality else None
    quality_display = f"{quality_percent}%" if quality_percent is not None else "N/A"
    print(f"🎯 Final: quality {quality_display}, {output.iteration_count} iterations")
    return output.summary


def stream_summarize_video(
    transcript_or_url: str,
    target_language: str | None = None,
) -> Generator[SummarizerState, None, None]:
    """Stream the summarization process with progress updates."""
    graph = create_graph()
    transcript = _extract_transcript(transcript_or_url)

    initial_state = SummarizerState(
        transcript=transcript,
        target_language=target_language or TARGET_LANGUAGE,
    )
    for chunk in graph.stream(initial_state.model_dump(), stream_mode="values"):
        yield SummarizerState.model_validate(chunk)
