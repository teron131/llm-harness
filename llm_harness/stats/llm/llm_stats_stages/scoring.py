"""Score helpers for native Python LLM stats."""

from __future__ import annotations

from statistics import median
from typing import Any

from ..shared import as_finite_number, as_record


def percentile_rank(values: list[Any], value: Any) -> float | None:
    numeric_value = as_finite_number(value)
    if numeric_value is None:
        return None
    finite_values = [item for item in (as_finite_number(v) for v in values) if item is not None]
    if not finite_values:
        return None
    return (sum(1 for item in finite_values if item <= numeric_value) / len(finite_values)) * 100


def mean_of_finite(values: list[float | None]) -> float | None:
    finite_values = [value for value in values if value is not None]
    return (sum(finite_values) / len(finite_values)) if finite_values else None


def _metric_value(model: dict[str, Any], key: str) -> float | None:
    intelligence = as_record(model.get("intelligence"))
    evaluations = as_record(model.get("evaluations"))
    if key == "omniscience_nonhallucination_rate":
        return as_finite_number(intelligence.get("omniscience_nonhallucination_rate")) or as_finite_number(intelligence.get("omniscience_hallucination_rate"))
    return as_finite_number(intelligence.get(key)) or as_finite_number(evaluations.get(key))


def blended_price_value(cost_like: Any, scoring_config: dict[str, Any]) -> float | None:
    cost = as_record(cost_like)
    input_cost = as_finite_number(cost.get("input"))
    output_cost = as_finite_number(cost.get("output"))
    weighted_input = as_finite_number(cost.get("weighted_input"))
    weighted_output = as_finite_number(cost.get("weighted_output"))
    cache_read = as_finite_number(cost.get("cache_read"))
    cache_write = as_finite_number(cost.get("cache_write"))
    if input_cost is None or output_cost is None or input_cost <= 0 or output_cost <= 0:
        return None
    input_ratio = scoring_config.get("weighted_price_input_ratio", 0.75)
    output_ratio = scoring_config.get("weighted_price_output_ratio", 0.25)
    if weighted_input is not None or weighted_output is not None:
        effective_input = weighted_input if weighted_input is not None else input_cost
        effective_output = weighted_output if weighted_output is not None else output_cost
        return input_ratio * effective_input + output_ratio * effective_output
    cache_weighted_input = cache_read if cache_read is not None else input_cost
    cache_weighted_output = 0.1 * cache_write + 0.9 * output_cost if cache_write is not None else output_cost
    return input_ratio * (input_ratio * cache_weighted_input + output_ratio * input_cost) + output_ratio * cache_weighted_output


def derive_speed_output_token_anchors(openrouter_speed_by_id: dict[str, dict[str, Any]], scoring_config: dict[str, Any]) -> list[int]:
    implied_token_usages: list[float] = []
    for speed in openrouter_speed_by_id.values():
        throughput = as_finite_number(speed.get("throughput_tokens_per_second_median"))
        latency = as_finite_number(speed.get("latency_seconds_median"))
        e2e_latency = as_finite_number(speed.get("e2e_latency_seconds_median"))
        if throughput is None or throughput <= 0 or latency is None or e2e_latency is None:
            continue
        generation_seconds = e2e_latency - latency
        if generation_seconds <= 0:
            continue
        implied_token_usages.append(generation_seconds * throughput)
    implied_token_usages.sort()
    default_anchors = list(scoring_config.get("default_speed_output_token_anchors", [200, 500, 1000, 2000, 8000]))
    if not implied_token_usages:
        return default_anchors
    quantiles = [0, *scoring_config.get("speed_anchor_quantiles", [0.25, 0.5, 0.75]), 1]
    anchors: list[float] = []
    for quantile in quantiles:
        if len(implied_token_usages) == 1:
            anchors.append(implied_token_usages[0])
            continue
        index = (len(implied_token_usages) - 1) * quantile
        lower = int(index)
        upper = min(lower + 1, len(implied_token_usages) - 1)
        ratio = index - lower
        anchors.append(implied_token_usages[lower] + (implied_token_usages[upper] - implied_token_usages[lower]) * ratio)
    source_min = anchors[0]
    source_max = anchors[-1]
    if source_max <= source_min:
        return default_anchors
    range_min = scoring_config.get("speed_output_token_range_min", 200)
    range_max = scoring_config.get("speed_output_token_range_max", 8000)
    return [round(range_min + ((anchor - source_min) / (source_max - source_min)) * (range_max - range_min)) for anchor in anchors]


def build_scores(model: dict[str, Any], cost: Any, speed: dict[str, Any], speed_output_token_anchors: list[int], scoring_config: dict[str, Any]) -> dict[str, Any] | None:
    intelligence_index = _metric_value(model, "intelligence_index") or _metric_value(model, "artificial_analysis_intelligence_index")
    agentic_index = _metric_value(model, "agentic_index") or _metric_value(model, "artificial_analysis_agentic_index")
    intelligence_benchmark_mean = mean_of_finite([_metric_value(model, key) for key in scoring_config.get("intelligence_benchmark_keys", [])])
    agentic_benchmark_mean = mean_of_finite([_metric_value(model, key) for key in scoring_config.get("agentic_benchmark_keys", [])])
    intelligence_score = (intelligence_index + intelligence_benchmark_mean * 100) / 2 if intelligence_index is not None and intelligence_benchmark_mean is not None else None
    agentic_score = (agentic_index + agentic_benchmark_mean * 100) / 2 if agentic_index is not None and agentic_benchmark_mean is not None else None
    blended_price = blended_price_value(cost, scoring_config)
    latency = as_finite_number(speed.get("latency_seconds_median"))
    throughput = as_finite_number(speed.get("throughput_tokens_per_second_median"))
    e2e_latency = as_finite_number(speed.get("e2e_latency_seconds_median"))
    price_score = 1 / blended_price if blended_price is not None and blended_price > 0 else None
    imagined_speed_score = mean_of_finite(
        [target / (latency + target / throughput) if latency is not None and throughput is not None and throughput > 0 else None for target in speed_output_token_anchors]
    )
    representative_target_tokens = median(sorted(speed_output_token_anchors)) if speed_output_token_anchors else None
    observed_e2e_speed_score = representative_target_tokens / e2e_latency if representative_target_tokens is not None and e2e_latency is not None and e2e_latency > 0 else None
    speed_score = mean_of_finite([imagined_speed_score, observed_e2e_speed_score])
    if all(score is None for score in (intelligence_score, agentic_score, price_score, speed_score)):
        return None
    return {
        "intelligence_score": intelligence_score,
        "agentic_score": agentic_score,
        "speed_score": speed_score,
        "price_score": price_score,
    }


def attach_relative_scores(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def min_max_scale(values: list[float | None], value: float | None) -> float | None:
        if value is None:
            return None
        finite_values = [candidate for candidate in values if candidate is not None]
        if not finite_values:
            return None
        min_value = min(finite_values)
        max_value = max(finite_values)
        if max_value == min_value:
            return 100
        return ((value - min_value) / (max_value - min_value)) * 100

    intelligence_scores = [as_finite_number(as_record(model.get("scores")).get("intelligence_score")) for model in models]
    agentic_scores = [as_finite_number(as_record(model.get("scores")).get("agentic_score")) for model in models]
    speed_scores = [as_finite_number(as_record(model.get("scores")).get("speed_score")) for model in models]
    price_scores = [as_finite_number(as_record(model.get("scores")).get("price_score")) for model in models]
    output = []
    for model in models:
        scores = as_record(model.get("scores"))
        intelligence_score = as_finite_number(scores.get("intelligence_score"))
        agentic_score = as_finite_number(scores.get("agentic_score"))
        speed_score = as_finite_number(scores.get("speed_score"))
        price_score = as_finite_number(scores.get("price_score"))
        intelligence_relative = min_max_scale(intelligence_scores, intelligence_score)
        agentic_relative = min_max_scale(agentic_scores, agentic_score)
        speed_relative = percentile_rank(speed_scores, speed_score)
        price_relative = percentile_rank(price_scores, price_score)
        overall_relative = mean_of_finite([intelligence_relative, agentic_relative, speed_relative, price_relative])
        output.append(
            {
                **model,
                "relative_scores": {
                    "intelligence_score": intelligence_relative,
                    "agentic_score": agentic_relative,
                    "speed_score": speed_relative,
                    "price_score": price_relative,
                    "overall_score": overall_relative,
                },
            }
        )
    return output
