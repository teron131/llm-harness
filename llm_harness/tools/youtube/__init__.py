"""YouTubeLoader - Load and extract transcript from YouTube videos using Scrape Creators API."""

from .scrapper import scrape_youtube
from .summarizer import stream_summarize_video, summarize_video


def youtube_loader(url: str) -> str:
    """Load and extract transcript from a YouTube video.

    Args:
        url: YouTube video URL

    Returns:
        Formatted string containing video metadata and transcript

    Raises:
        ValueError: If URL is invalid or API key is missing
        requests.RequestException: If API request fails
    """
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
    "youtube_loader",
]
