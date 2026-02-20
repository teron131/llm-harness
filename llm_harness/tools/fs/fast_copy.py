"""General-purpose utilities for line-tagged content processing.

These utilities support adding, removing, and filtering content using line-level
tags (e.g., [L1], [L2]). This is useful for LLM workflows where models identify
specific sections of text by their line numbers.
"""

import re

from pydantic import BaseModel, Field


class TagRange(BaseModel):
    """Represents a range of lines to be removed from the content."""

    start_tag: str = Field(description="The starting line tag, e.g., [L10]")
    end_tag: str = Field(description="The ending line tag, e.g., [L20]")


def tag_content(text: str) -> str:
    """Prepend [LX] tags to each line of the content."""
    lines = text.splitlines()
    return "\n".join(f"[L{i + 1}] {line}" for i, line in enumerate(lines))


def untag_content(text: str) -> str:
    """Remove [LX] tags from the content."""
    # Use re.MULTILINE to match at the start of each line
    return re.sub(r"^\[L\d+\]\s*", "", text, flags=re.MULTILINE)


def filter_content(tagged_text: str, ranges: list[TagRange]) -> str:
    """Remove lines between start_tag and end_tag from the tagged content."""
    lines = tagged_text.splitlines()
    if not lines or not ranges:
        return tagged_text

    # 1. Build tag mapping efficiently
    tag_to_idx: dict[str, int] = {}
    for line_idx, line in enumerate(lines):
        if line.startswith("[L"):
            end_bracket = line.find("]")
            if end_bracket != -1:
                tag_to_idx[line[: end_bracket + 1]] = line_idx

    # 2. Boolean mask (initialized to True = keep)
    keep_mask = [True] * len(lines)

    # 3. Mark ranges to remove
    for tag_range in ranges:
        start_idx = tag_to_idx.get(tag_range.start_tag)
        end_idx = tag_to_idx.get(tag_range.end_tag)
        if start_idx is not None and end_idx is not None:
            # Ensure correct ordering and inclusive range
            first_idx, last_idx = sorted((start_idx, end_idx))
            keep_mask[first_idx : last_idx + 1] = [False] * (last_idx - first_idx + 1)

    # 4. Filter and join
    return "\n".join(line for line, keep in zip(lines, keep_mask, strict=True) if keep)
