"""Shared helpers for native Python LLM stats modules."""

from __future__ import annotations

import re
from typing import Any

from ..utils import JsonObject, as_finite_number, as_record

__all__ = [
    "FALLBACK_PROVIDER_IDS",
    "PRIMARY_PROVIDER_ID",
    "JsonObject",
    "as_finite_number",
    "as_record",
    "model_slug_from_model_id",
    "normalize_model_token",
    "normalize_provider_model_id",
]

PRIMARY_PROVIDER_ID = "openrouter"
FALLBACK_PROVIDER_IDS = {"openai", "google", "anthropic"}


def normalize_model_token(value: str) -> str:
    normalized = value.lower()
    normalized = re.sub(r"[._:\s]+", "-", normalized)
    normalized = re.sub(r"[^a-z0-9/-]+", "", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-/")


def model_slug_from_model_id(model_id: Any) -> str | None:
    if not isinstance(model_id, str) or not model_id:
        return None
    slug = model_id.split("/")[-1]
    return slug or None


def normalize_provider_model_id(model_id: str) -> str:
    slash_index = model_id.find("/")
    if slash_index <= 0:
        return re.sub(r"-+", "-", model_id.lower().replace(".", "-"))
    provider = model_id[:slash_index].lower()
    base_model_id = re.sub(
        r"-+",
        "-",
        model_id[slash_index + 1 :].lower().replace(".", "-"),
    )
    return f"{provider}/{base_model_id}"
