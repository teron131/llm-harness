"""Cache helpers for the final selected LLM stats payload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...utils import is_fresh_epoch_seconds, now_epoch_seconds, write_json_file
from .types import ModelStatsSelectedPayload, ModelStatsSelectedPayloadModel

DEFAULT_OUTPUT_PATH = Path(".cache/llm_stats.json")
CACHE_TTL_SECONDS = 60 * 60 * 24


def current_epoch_seconds() -> int:
    return now_epoch_seconds()


def save_model_stats_selected_to_path(
    payload: ModelStatsSelectedPayload,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    try:
        validated_payload = ModelStatsSelectedPayloadModel.model_validate(payload)
        write_json_file(output_path, validated_payload.model_dump())
    except Exception:
        return


def load_model_stats_selected_from_cache(
    output_path: Path,
) -> ModelStatsSelectedPayload | None:
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
        return ModelStatsSelectedPayloadModel(
            fetched_at_epoch_seconds=int(fetched_at_epoch_seconds),
            models=models,
        ).model_dump()
    except Exception:
        return None
