"""Public LLM stats API with cache-first list mode and staged orchestration."""

from __future__ import annotations

from pathlib import Path

from .llm_stats_stages.cache import (
    DEFAULT_OUTPUT_PATH,
    current_epoch_seconds,
    load_model_stats_selected_from_cache,
    save_model_stats_selected_to_path,
)
from .llm_stats_stages.final_stage import build_final_payload
from .llm_stats_stages.match_stage import build_matched_payload
from .llm_stats_stages.openrouter_stage import enrich_rows
from .llm_stats_stages.source_stage import fetch_source_data
from .llm_stats_stages.types import (
    LlmSourceData,
    LlmStatsStageConfig,
    LlmStatsStageConfigModel,
    ModelStatsSelectedOptions,
    ModelStatsSelectedOptionsModel,
    ModelStatsSelectedPayload,
    ModelStatsSelectedPayloadModel,
)

LLM_STATS_STAGE_CONFIG_MODEL = LlmStatsStageConfigModel(
    matcher={
        "variant_tokens": ["flash-lite", "flash", "pro", "nano", "mini", "lite"],
    },
    openrouter={
        "speed_concurrency": 8,
    },
    final={
        "null_field_prune_threshold": 0.5,
        "null_field_prune_recent_lookback_days": 90,
    },
    scoring={
        "intelligence_benchmark_keys": [
            "omniscience_accuracy",
            "hle",
            "lcr",
            "scicode",
        ],
        "agentic_benchmark_keys": [
            "omniscience_nonhallucination_rate",
            "gdpval_normalized",
            "ifbench",
            "terminalbench_hard",
        ],
        "default_speed_output_token_anchors": [200, 500, 1_000, 2_000, 8_000],
        "speed_output_token_range_min": 200,
        "speed_output_token_range_max": 8_000,
        "speed_anchor_quantiles": [0.25, 0.5, 0.75],
        "weighted_price_input_ratio": 0.75,
        "weighted_price_output_ratio": 0.25,
    },
)
LLM_STATS_STAGE_CONFIG: LlmStatsStageConfig = LLM_STATS_STAGE_CONFIG_MODEL.model_dump()


def save_model_stats_selected(
    payload: ModelStatsSelectedPayload,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    validated_payload = ModelStatsSelectedPayloadModel.model_validate(payload)
    save_model_stats_selected_to_path(validated_payload.model_dump(), output_path)


def get_model_stats_selected(
    options: ModelStatsSelectedOptions | None = None,
) -> ModelStatsSelectedPayload:
    """Build the final selected LLM stats payload with cache-first list mode."""
    options_model = ModelStatsSelectedOptionsModel.model_validate(options or {})
    model_id = options_model.id
    try:
        if model_id is None:
            cached_payload = load_model_stats_selected_from_cache(DEFAULT_OUTPUT_PATH)
            if cached_payload is not None:
                return ModelStatsSelectedPayloadModel.model_validate(cached_payload).model_dump()
        source_data = fetch_source_data()
        matched_payload = build_matched_payload(
            source_data,
            LLM_STATS_STAGE_CONFIG.get("matcher"),
        )
        enriched = enrich_rows(
            matched_payload.get("models") or [],
            LLM_STATS_STAGE_CONFIG.get("openrouter"),
            LLM_STATS_STAGE_CONFIG.get("scoring"),
        )
        payload = build_final_payload(
            enriched.get("rows") or [],
            model_id=model_id if isinstance(model_id, str) else None,
            fetched_at_epoch_seconds=current_epoch_seconds(),
            openrouter_speed_by_id=enriched.get("openrouter_speed_by_id") or {},
            openrouter_pricing_by_id=enriched.get("openrouter_pricing_by_id") or {},
            speed_output_token_anchors=enriched.get("speed_output_token_anchors") or [],
            scoring_config=LLM_STATS_STAGE_CONFIG.get("scoring"),
        )
        validated_payload = ModelStatsSelectedPayloadModel.model_validate(payload)
        if model_id is not None:
            return validated_payload.model_dump()
        save_model_stats_selected(validated_payload.model_dump(), DEFAULT_OUTPUT_PATH)
        return validated_payload.model_dump()
    except Exception:
        return ModelStatsSelectedPayloadModel(
            fetched_at_epoch_seconds=None,
            models=[],
        ).model_dump()


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "LLM_STATS_STAGE_CONFIG",
    "LLM_STATS_STAGE_CONFIG_MODEL",
    "LlmSourceData",
    "LlmStatsStageConfig",
    "ModelStatsSelectedOptions",
    "ModelStatsSelectedPayload",
    "get_model_stats_selected",
    "save_model_stats_selected",
]
