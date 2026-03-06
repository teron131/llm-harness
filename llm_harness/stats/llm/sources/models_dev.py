"""models.dev LLM source ported from the JS stats pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, TypedDict

import httpx

MODELS_DEV_URL = "https://models.dev/api.json"
LOOKBACK_DAYS = 365
REQUEST_TIMEOUT_SECONDS = 30.0


class ModelsDevOptions(TypedDict):
    """Reserved options for future extension."""


class ModelsDevOutputPayload(TypedDict):
    fetched_at_epoch_seconds: int | None
    status_code: int | None
    models: list[dict[str, Any]]


def _now_epoch_seconds() -> int:
    return int(datetime.now(UTC).timestamp())


def _iso_date_days_ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).date().isoformat()


def _is_recent_date(iso_date: str | None, cutoff_iso_date: str) -> bool:
    return bool(iso_date) and iso_date >= cutoff_iso_date


def _as_finite_number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def _fetch_models_dev() -> dict[str, Any]:
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = client.get(MODELS_DEV_URL)
    response.raise_for_status()
    return {
        "fetched_at_epoch_seconds": _now_epoch_seconds(),
        "status_code": response.status_code,
        "payload": response.json(),
    }


def _flatten_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider_id, provider in payload.items():
        if not isinstance(provider, dict):
            continue
        provider_name = provider.get("name") or provider_id
        models = provider.get("models") or {}
        if not isinstance(models, dict):
            continue
        for model_id, model in models.items():
            if not isinstance(model, dict):
                continue
            rows.append(
                {
                    "provider_id": provider_id,
                    "provider_name": provider_name,
                    "model_id": model.get("id") or model_id,
                    "model": model,
                }
            )
    return rows


def _rank_recent_models(
    models: list[dict[str, Any]],
    cutoff_iso_date: str,
) -> list[dict[str, Any]]:
    filtered = [row for row in models if _is_recent_date((row.get("model") or {}).get("release_date"), cutoff_iso_date)]
    return sorted(
        filtered,
        key=lambda row: (
            _as_finite_number(((row.get("model") or {}).get("cost") or {}).get("output"))
            if _as_finite_number(((row.get("model") or {}).get("cost") or {}).get("output")) is not None
            else float("inf"),
            -ord("a"),
        ),
    )


def get_models_dev_stats(
    _options: ModelsDevOptions | None = None,
) -> ModelsDevOutputPayload:
    """Fetch, flatten, and rank recent models from models.dev."""
    try:
        source_payload = _fetch_models_dev()
        cutoff_iso_date = _iso_date_days_ago(LOOKBACK_DAYS)
        ranked = _rank_recent_models(
            _flatten_models(source_payload.get("payload") or {}),
            cutoff_iso_date,
        )
        ranked.sort(
            key=lambda row: (row.get("model") or {}).get("release_date") or "",
            reverse=True,
        )
        ranked.sort(
            key=lambda row: (
                _as_finite_number(((row.get("model") or {}).get("cost") or {}).get("output"))
                if _as_finite_number(((row.get("model") or {}).get("cost") or {}).get("output")) is not None
                else float("inf")
            ),
        )
        return {
            "fetched_at_epoch_seconds": source_payload["fetched_at_epoch_seconds"],
            "status_code": source_payload["status_code"],
            "models": ranked,
        }
    except Exception:
        return {
            "fetched_at_epoch_seconds": None,
            "status_code": None,
            "models": [],
        }


__all__ = [
    "ModelsDevOptions",
    "get_models_dev_stats",
]
