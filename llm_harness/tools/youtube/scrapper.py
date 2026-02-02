"""This module uses the Scrape Creators or Supadata API to scrape a YouTube video transcript.


The API result is wrapped by YouTubeScrapperResult object.
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
import requests

from .utils import clean_text, clean_youtube_url, is_youtube_url

load_dotenv()

SCRAPECREATORS_API_KEY = os.getenv("SCRAPECREATORS_API_KEY")
SUPADATA_API_KEY = os.getenv("SUPADATA_API_KEY")

# API Endpoints
SCRAPECREATORS_ENDPOINT = "https://api.scrapecreators.com/v1/youtube/video/transcript"
SUPADATA_ENDPOINT = "https://api.supadata.ai/v1/transcript"


def _extract_video_id(url: str) -> str | None:
    """Extract video ID from YouTube URL."""
    import re

    # Standard watch URL
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    # Short URL
    match = re.search(r"youtu\.be/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    return None


class Channel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    url: str | None = None
    handle: str | None = None
    title: str | None = None


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str | None = None
    startMs: float | None = None
    endMs: float | None = None
    startTimeText: str | None = None


class YouTubeScrapperResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    success: bool | None = None
    credits_remaining: float | None = None
    type: str | None = None
    transcript: list[TranscriptSegment] | None = None
    transcript_only_text: str | None = None
    title: str | None = None
    description: str | None = None
    thumbnail: str | None = None
    url: str | None = None
    id: str | None = None
    viewCountInt: int | None = None
    likeCountInt: int | None = None
    publishDate: str | None = None
    channel: Channel | None = None
    durationFormatted: str | None = None
    keywords: list[str] | None = None
    videoId: str | None = None
    captionTracks: list[dict] | None = None
    language: str | None = None

    @property
    def parsed_transcript(self) -> str | None:
        """Return cleaned transcript text or None if unavailable."""
        if self.transcript:
            return clean_text(" ".join([seg.text for seg in self.transcript if seg.text]))
        if self.transcript_only_text and self.transcript_only_text.strip():
            return clean_text(self.transcript_only_text)
        return None

    @property
    def has_transcript(self) -> bool:
        """Check if video has a transcript available."""
        return bool(self.transcript or (self.transcript_only_text and self.transcript_only_text.strip()))


def _fetch_scrape_creators(video_url: str) -> YouTubeScrapperResult | None:
    """Fetch transcript from Scrape Creators API."""
    if not SCRAPECREATORS_API_KEY:
        return None

    try:
        url = f"{SCRAPECREATORS_ENDPOINT}?url={video_url}"
        headers = {"x-api-key": SCRAPECREATORS_API_KEY}
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code in [401, 403]:
            print("Scrape Creators API auth failed")
            return None

        response.raise_for_status()
        data = response.json()

        # Normalize transcript segments
        if "transcript" in data and isinstance(data["transcript"], list):
            for seg in data["transcript"]:
                if "startMs" in seg:
                    seg["startMs"] = float(seg["startMs"])
                if "endMs" in seg:
                    seg["endMs"] = float(seg["endMs"])

        return YouTubeScrapperResult.model_validate(data)
    except Exception as e:
        print(f"Scrape Creators fetch error: {e}")
        return None


def _fetch_supadata(video_url: str) -> YouTubeScrapperResult | None:
    """Fetch transcript from Supadata API."""
    if not SUPADATA_API_KEY:
        return None

    try:
        url = f"{SUPADATA_ENDPOINT}?url={video_url}&lang=en&text=true"
        headers = {"x-api-key": SUPADATA_API_KEY}
        response = requests.get(url, headers=headers, timeout=30)

        if not response.ok:
            print(f"Supadata API error: {response.status_code}")
            return None

        data = response.json()
        content = data.get("content", [])

        # Normalize to shared schema
        transcript = []
        for item in content:
            offset_ms = item.get("offset", 0)
            duration_ms = item.get("duration", 0)
            transcript.append(
                {
                    "text": item.get("text"),
                    "startMs": offset_ms,
                    "endMs": offset_ms + duration_ms,
                    # Simple formatting: convert ms to seconds string with 's' suffix
                    "startTimeText": f"{offset_ms / 1000:.2f}s",
                }
            )

        video_id = _extract_video_id(video_url)

        return YouTubeScrapperResult(
            url=video_url,
            transcript=[TranscriptSegment.model_validate(t) for t in transcript],
            videoId=video_id,
            language=data.get("lang"),
            success=True,
            type="video",
        )
    except Exception as e:
        print(f"Supadata fetch error: {e}")
        return None


def scrape_youtube(youtube_url: str) -> YouTubeScrapperResult:
    """Scrape a YouTube video and return the transcript.

    Tries Scrape Creators first, then Supadata.

    Args:
        youtube_url: The YouTube video URL to scrape

    Returns:
        YouTubeScrapperResult: Parsed video data including transcript and metadata
    """
    if not is_youtube_url(youtube_url):
        raise ValueError("Invalid YouTube URL")

    youtube_url = clean_youtube_url(youtube_url)

    # Try Scrape Creators first
    result = _fetch_scrape_creators(youtube_url)
    if result and result.has_transcript:
        return result

    # Fallback to Supadata
    result = _fetch_supadata(youtube_url)
    if result and result.has_transcript:
        return result

    if not SCRAPECREATORS_API_KEY and not SUPADATA_API_KEY:
        raise ValueError("No API keys found for Scrape Creators or Supadata")

    raise ValueError("Failed to fetch transcript from available providers")
