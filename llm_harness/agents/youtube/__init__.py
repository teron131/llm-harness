"""YouTubeLoader - Load and extract transcript from YouTube videos using Scrape Creators API."""

from .schemas import Summary


def summarize_video(
    transcript_or_url: str,
    target_language: str | None = None,
) -> Summary:
    """Summarize a YouTube URL or transcript."""
    from .summarizer import summarize_video as _summarize_video

    return _summarize_video(
        transcript_or_url=transcript_or_url,
        target_language=target_language,
    )


def stream_summarize_video(
    transcript_or_url: str,
    target_language: str | None = None,
):
    """Stream summary generation for a YouTube URL or transcript."""
    from .summarizer_react import stream_summarize_video_react as _stream_summarize_video

    return _stream_summarize_video(
        transcript_or_url=transcript_or_url,
        target_language=target_language,
    )


def summarize_video_react(
    transcript_or_url: str,
    target_language: str | None = None,
):
    """Summarize a YouTube URL or transcript with the ReAct workflow."""
    from .summarizer_react import summarize_video_react as _summarize_video_react

    return _summarize_video_react(
        transcript_or_url=transcript_or_url,
        target_language=target_language,
    )


def youtube_loader(url: str) -> str:
    """Load and extract transcript from a YouTube video.

    Args:
        url: YouTube video URL

    Returns:
        Formatted string containing video metadata and transcript

    Raises:
        ValueError: If URL is invalid or API key is missing
        httpx.RequestError: If API request fails
    """
    from ...tools.youtube.scraper import scrape_youtube

    result = scrape_youtube(url)

    # Build formatted output
    output_parts = []

    # Video metadata
    if result.title:
        output_parts.append(f"Title: {result.title}")
    if result.channel and result.channel.title:
        output_parts.append(f"Channel: {result.channel.title}")
    if result.durationFormatted:
        output_parts.append(f"Duration: {result.durationFormatted}")
    if result.publishDateText:
        output_parts.append(f"Published: {result.publishDateText}")
    if result.viewCountInt is not None:
        output_parts.append(f"Views: {result.viewCountInt:,}")
    if result.likeCountInt is not None:
        output_parts.append(f"Likes: {result.likeCountInt:,}")

    output_parts.append("")  # Empty line separator

    # Description
    if result.description:
        output_parts.append(f"Description:\n{result.description}")
        output_parts.append("")

    # Transcript
    if result.has_transcript:
        transcript = result.parsed_transcript
        if transcript:
            output_parts.append(f"Transcript:\n{transcript}")
    else:
        output_parts.append("Transcript: Not available for this video")

    return "\n".join(output_parts)


__all__ = [
    "stream_summarize_video",
    "summarize_video",
    "summarize_video_react",
    "youtube_loader",
]
