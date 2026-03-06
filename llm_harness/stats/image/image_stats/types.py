"""Types for the staged Python image stats pipeline."""

from __future__ import annotations

from typing import Any, TypedDict


class ImageStatsSelectedOptions(TypedDict, total=False):
    id: str | None


class ImageStatsSelectedPayload(TypedDict):
    fetched_at_epoch_seconds: int | None
    models: list[dict[str, Any]]


class ImageSourceData(TypedDict):
    artificial_analysis_payload: dict[str, Any]
    arena_payload: dict[str, Any]
    artificial_analysis_models_by_slug: dict[str, dict[str, Any]]
    arena_models_by_name: dict[str, dict[str, Any]]


class ImageUnionRow(TypedDict):
    artificial_analysis_slug: str | None
    artificial_analysis_name: str | None
    artificial_analysis_provider: str | None
    best_match: dict[str, Any] | None
    candidates: list[dict[str, Any]]
    artificial_analysis: dict[str, Any] | None
    arena_ai: dict[str, Any] | None
