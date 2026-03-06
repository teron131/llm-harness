"""Shared types for the native Python LLM stats stages."""

from __future__ import annotations

from typing import Any, TypedDict


class LlmStatsStageConfig(TypedDict, total=False):
    matcher: dict[str, Any]
    openrouter: dict[str, Any]
    final: dict[str, Any]
    scoring: dict[str, Any]


class ModelStatsSelectedModel(TypedDict, total=False):
    id: str | None
    name: str | None
    provider: str | None
    logo: str
    attachment: bool | None
    reasoning: bool | None
    release_date: str | None
    modalities: Any
    open_weights: bool | None
    cost: Any
    context_window: Any
    speed: dict[str, Any]
    intelligence: Any
    intelligence_index_cost: Any
    evaluations: Any
    scores: Any
    relative_scores: Any


class ModelStatsSelectedPayload(TypedDict):
    fetched_at_epoch_seconds: int | None
    models: list[ModelStatsSelectedModel]


class ModelStatsSelectedOptions(TypedDict, total=False):
    id: str | None


class LlmSourceData(TypedDict):
    artificial_analysis_payload: dict[str, Any]
    scraped_payload: dict[str, Any]
    models_dev_payload: dict[str, Any]
    scraped_rows: list[dict[str, Any]]
    preferred_models_dev_models: list[dict[str, Any]]
    models_dev_by_id: dict[str, dict[str, Any]]
    api_by_slug: dict[str, dict[str, Any]]
    scraped_by_slug: dict[str, dict[str, Any]]


class EnrichedRows(TypedDict):
    rows: list[dict[str, Any]]
    openrouter_speed_by_id: dict[str, dict[str, Any]]
    openrouter_pricing_by_id: dict[str, dict[str, Any]]
    speed_output_token_anchors: list[int]
