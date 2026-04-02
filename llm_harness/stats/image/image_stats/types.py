"""Types for the staged Python image stats pipeline."""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class ImageStatsSelectedOptions(TypedDict, total=False):
    """Options for Typed models for image stats selection."""

    id: str | None


class ImageStatsSelectedPayload(TypedDict):
    """Payload for Typed models for image stats selection."""

    fetched_at_epoch_seconds: int | None
    models: list[dict[str, Any]]


class ImageSourceData(TypedDict):
    """Image Source Data for Typed models for image stats selection."""

    artificial_analysis_payload: dict[str, Any]
    arena_payload: dict[str, Any]
    artificial_analysis_models_by_slug: dict[str, dict[str, Any]]
    arena_models_by_name: dict[str, dict[str, Any]]


class ImageUnionRow(TypedDict):
    """Row model for Typed models for image stats selection."""

    artificial_analysis_slug: str | None
    artificial_analysis_name: str | None
    artificial_analysis_provider: str | None
    best_match: dict[str, Any] | None
    candidates: list[dict[str, Any]]
    artificial_analysis: dict[str, Any] | None
    arena_ai: dict[str, Any] | None


class ImageStatsSelectedOptionsModel(BaseModel):
    """Pydantic model for Typed models for image stats selection."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None


class ImageStatsSelectedModelModel(BaseModel):
    """Pydantic model for Typed models for image stats selection."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str | None = None
    provider: str | None = None
    logo: str = ""
    release_date: str | None = None
    sources: dict[str, Any] = Field(default_factory=dict)
    source_scores: dict[str, Any] = Field(default_factory=dict)
    source_percentiles: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, Any] = Field(default_factory=dict)
    percentiles: dict[str, Any] = Field(default_factory=dict)


class ImageStatsSelectedPayloadModel(BaseModel):
    """Pydantic model for Typed models for image stats selection."""

    model_config = ConfigDict(extra="forbid")

    fetched_at_epoch_seconds: int | None = None
    models: list[ImageStatsSelectedModelModel] = Field(default_factory=list)
