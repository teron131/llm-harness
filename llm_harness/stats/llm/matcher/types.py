"""Shared public and intermediate types for the native Python LLM matcher."""

from __future__ import annotations

from typing import Any, TypedDict


class MatcherSourceModel(TypedDict):
    """Pydantic model for Typed models for LLM matching."""

    source_slug: str
    source_name: str | None
    source_release_date: str | None


class LlmMatchCandidate(TypedDict):
    """Candidate model for Typed models for LLM matching."""

    model_id: str
    provider_id: str
    provider_name: str
    model_name: str | None
    score: float


class LlmMatchMappedModel(TypedDict):
    """Pydantic model for Typed models for LLM matching."""

    artificial_analysis_slug: str
    artificial_analysis_name: str | None
    artificial_analysis_release_date: str | None
    best_match: LlmMatchCandidate | None
    candidates: list[LlmMatchCandidate]


class LlmMatchModelMappingPayload(TypedDict):
    """Payload for Typed models for LLM matching."""

    artificial_analysis_fetched_at_epoch_seconds: int | None
    models_dev_fetched_at_epoch_seconds: int | None
    total_artificial_analysis_models: int
    total_models_dev_models: int
    max_candidates: int
    void_mode: str
    void_threshold: float | None
    voided_count: int
    models: list[LlmMatchMappedModel]


class LlmScraperFallbackMatchDiagnosticsPayload(TypedDict):
    """Payload for Typed models for LLM matching."""

    scraped_fetched_at_epoch_seconds: int | None
    models_dev_fetched_at_epoch_seconds: int | None
    total_scraped_models: int
    total_models_dev_models: int
    max_candidates: int
    pre_void_matched_count: int
    pre_void_unmatched_count: int
    void_mode: str
    void_threshold: float | None
    voided_count: int
    matched_count: int
    unmatched_count: int
    models: list[LlmMatchMappedModel]


class LlmMatchModelMappingOptions(TypedDict, total=False):
    """Options for Typed models for LLM matching."""

    max_candidates: int
    artificial_analysis_models: list[dict[str, Any]]
    models_dev_models: list[dict[str, Any]]
    scraped_rows: list[dict[str, Any]]


class PreferredProviderPools(TypedDict):
    """Preferred Provider Pools for Typed models for LLM matching."""

    primary: list[dict[str, Any]]
    fallback: list[dict[str, Any]]


class MatcherRunOutput(TypedDict):
    """Matcher Run Output for Typed models for LLM matching."""

    models: list[LlmMatchMappedModel]
    void_threshold: float | None
    voided_count: int
    pre_void_matched_count: int
    pre_void_unmatched_count: int
    matched_count: int
    unmatched_count: int
