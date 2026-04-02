"""Shared YouTube text and URL utilities."""

from __future__ import annotations

import re

YOUTUBE_PATTERNS = (
    r"youtube\.com/watch\?v=",
    r"youtu\.be/",
    r"youtube\.com/embed/",
    r"youtube\.com/v/",
)


def clean_text(text: str) -> str:
    """Clean text by removing excessive whitespace and normalizing."""
    text = re.sub(r"\n{3,}", r"\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def clean_youtube_url(url: str) -> str:
    """Normalize YouTube URL variants to watch URL when possible."""
    if "youtube.com/watch" in url:
        match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
    elif "youtu.be/" in url:
        match = re.search(r"youtu\.be/([a-zA-Z0-9_-]+)", url)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
    return url


def is_youtube_url(url: str) -> bool:
    """Check if a URL is a valid YouTube URL."""
    return any(re.search(pattern, url) for pattern in YOUTUBE_PATTERNS)


def extract_video_id(url: str) -> str | None:
    """Extract video ID from YouTube URL."""
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    match = re.search(r"youtu\.be/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    return None
