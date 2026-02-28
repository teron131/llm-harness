"""Final selected model stats API (cache-first list mode, in-memory id mode)."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, TypedDict

from .data_sources.matcher import get_match_models_union

DEFAULT_OUTPUT_PATH = Path(".cache/model_stats.json")
CACHE_TTL_SECONDS = 60 * 60 * 24


class ModelStatsSelectedOptions(TypedDict, total=False):
    """Options for final model stats lookup."""

    id: str | None


class ModelStatsSelectedPayload(TypedDict):
    """Final selected model stats payload."""

    fetched_at_epoch_seconds: int | None
    models: list[dict[str, Any]]


def _provider_from_id(model_id: Any) -> str | None:
    if not isinstance(model_id, str):
        return None
    slash_index = model_id.find("/")
    if slash_index <= 0:
        return None
    return model_id[:slash_index]


def _now_epoch_seconds() -> int:
    return int(datetime.now(UTC).timestamp())


def _is_fresh_cache(fetched_at_epoch_seconds: Any) -> bool:
    if not isinstance(fetched_at_epoch_seconds, (int, float)):
        return False
    age_seconds = _now_epoch_seconds() - int(fetched_at_epoch_seconds)
    return 0 <= age_seconds <= CACHE_TTL_SECONDS


def _filter_models_by_id(models: list[dict[str, Any]], model_id: str | None) -> list[dict[str, Any]]:
    if model_id is None:
        return models
    return [model for model in models if model.get("id") == model_id]


def _build_logo(model: dict[str, Any], provider: str | None) -> str:
    model_creator = model.get("model_creator") or {}
    logo_slug = model_creator.get("slug") if isinstance(model_creator, dict) else None
    if isinstance(logo_slug, str) and logo_slug:
        return f"https://artificialanalysis.ai/img/logos/{logo_slug}_small.svg"
    provider_name = provider or "unknown"
    return f"https://models.dev/logos/{provider_name}.svg"


def _build_speed(model: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in model.items() if key.startswith("median_")}


def _map_union_model_to_selected(union_model: dict[str, Any]) -> dict[str, Any]:
    provider = _provider_from_id(union_model.get("id"))
    return {
        "id": union_model.get("id") if isinstance(union_model.get("id"), str) else None,
        "name": union_model.get("name") if isinstance(union_model.get("name"), str) else None,
        "provider": provider,
        "logo": _build_logo(union_model, provider),
        "attachment": union_model.get("attachment") if isinstance(union_model.get("attachment"), bool) else None,
        "reasoning": union_model.get("reasoning") if isinstance(union_model.get("reasoning"), bool) else None,
        "release_date": union_model.get("release_date") if isinstance(union_model.get("release_date"), str) else None,
        "modalities": union_model.get("modalities"),
        "open_weights": union_model.get("open_weights") if isinstance(union_model.get("open_weights"), bool) else None,
        "cost": union_model.get("cost"),
        "context_window": union_model.get("limit"),
        "speed": _build_speed(union_model),
        "evaluations": union_model.get("evaluations"),
        "scores": union_model.get("scores"),
        "percentiles": union_model.get("percentiles"),
    }


def save_model_stats_selected(
    payload: ModelStatsSelectedPayload,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Persist selected model stats payload to disk."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    except Exception:
        # Intentionally swallow cache write failures: API remains in-memory first.
        return


def _load_model_stats_selected_from_cache(
    output_path: Path,
) -> ModelStatsSelectedPayload | None:
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        models = payload.get("models")
        if not isinstance(models, list):
            return None
        fetched_at_epoch_seconds = payload.get("fetched_at_epoch_seconds")
        if not _is_fresh_cache(fetched_at_epoch_seconds):
            return None
        return {
            "fetched_at_epoch_seconds": int(fetched_at_epoch_seconds),
            "models": models,
        }
    except Exception:
        return None


def get_model_stats_selected(
    options: ModelStatsSelectedOptions | None = None,
) -> ModelStatsSelectedPayload:
    """Return final model stats from matcher union data."""
    options = options or {}
    model_id = options.get("id")

    try:
        if model_id is None:
            cached_payload = _load_model_stats_selected_from_cache(DEFAULT_OUTPUT_PATH)
            if cached_payload is not None:
                return cached_payload

        match_union = get_match_models_union()
        all_models = [_map_union_model_to_selected(union_model) for union_model in match_union.get("models") or [] if isinstance(union_model, dict)]
        filtered_models = _filter_models_by_id(all_models, model_id)
        fetched_at = _now_epoch_seconds()

        if model_id is not None:
            return {
                "fetched_at_epoch_seconds": fetched_at,
                "models": filtered_models,
            }

        list_payload: ModelStatsSelectedPayload = {
            "fetched_at_epoch_seconds": fetched_at,
            "models": filtered_models,
        }
        save_model_stats_selected(list_payload, DEFAULT_OUTPUT_PATH)
        return list_payload
    except Exception:
        return {
            "fetched_at_epoch_seconds": None,
            "models": [],
        }
