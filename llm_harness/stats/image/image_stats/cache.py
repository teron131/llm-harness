"""Cache helpers for the final selected image stats payload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...utils import is_fresh_epoch_seconds, now_epoch_seconds, write_json_file
from .types import ImageStatsSelectedPayload

DEFAULT_OUTPUT_PATH = Path(".cache/image_stats.json")
CACHE_TTL_SECONDS = 60 * 60 * 24


def current_epoch_seconds() -> int:
    return now_epoch_seconds()


def save_image_stats_selected_to_path(
    payload: ImageStatsSelectedPayload,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    try:
        write_json_file(output_path, payload)
    except Exception:
        return


def load_image_stats_selected_from_cache(
    output_path: Path,
) -> ImageStatsSelectedPayload | None:
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        models = payload.get("models")
        if not isinstance(models, list):
            return None
        fetched_at_epoch_seconds: Any = payload.get("fetched_at_epoch_seconds")
        if not is_fresh_epoch_seconds(fetched_at_epoch_seconds, CACHE_TTL_SECONDS):
            return None
        return {
            "fetched_at_epoch_seconds": int(fetched_at_epoch_seconds),
            "models": models,
        }
    except Exception:
        return None
