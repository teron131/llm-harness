"""Artificial Analysis model stats client with cache + scoring output."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import math
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import httpx
import polars as pl

load_dotenv()

MODELS_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
CACHE_PATH = Path(".cache/aa_models.json")
OUTPUT_PATH = Path(".cache/aa_output.json")
LOOKBACK_DAYS = 365
REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_CACHE_TTL_SECONDS = 60 * 60 * 12
BENCHMARK_BIAS_KEYS = ("hle", "terminalbench_hard", "lcr", "ifbench", "scicode")
PERCENTILE_COLUMNS = (
    ("overall_score", "overall_percentile"),
    ("intelligence_score", "intelligence_percentile"),
    ("speed_score", "speed_percentile"),
    ("price_score", "price_percentile"),
)


def _remove_ids(value: Any) -> Any:
    if isinstance(value, list):
        return [_remove_ids(item) for item in value]
    if isinstance(value, dict):
        return {key: _remove_ids(child) for key, child in value.items() if key != "id"}
    return value


def _as_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mean(values: list[float | None]) -> float | None:
    numbers = [value for value in values if value is not None and math.isfinite(value)]
    if not numbers:
        return None
    return float(sum(numbers) / len(numbers))


def _mean_positive(values: list[float | None]) -> float | None:
    numbers = [value for value in values if value is not None and math.isfinite(value) and value > 0]
    if not numbers:
        return None
    return float(sum(numbers) / len(numbers))


def _inverse_log1p(value: float) -> float:
    return float(1.0 / math.log1p(value))


def _load_cache() -> dict[str, Any]:
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fetch_and_cache_models(api_key: str | None) -> dict[str, Any]:
    if not api_key:
        msg = "Missing ARTIFICIALANALYSIS_API_KEY for refresh. Set it or use existing cache."
        raise ValueError(msg)

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = client.get(MODELS_URL, headers={"x-api-key": api_key})
    response.raise_for_status()

    payload = response.json()
    cache_payload = {
        "fetched_at_epoch_seconds": int(datetime.now(UTC).timestamp()),
        "status_code": response.status_code,
        "models": [_remove_ids(model) for model in payload.get("data", [])],
    }
    _write_json(CACHE_PATH, cache_payload)
    return cache_payload


def get_aa_stats() -> dict[str, Any]:
    """Return filtered + scored AA model stats JSON and write `.cache/aa_output.json`."""
    api_key = os.getenv("ARTIFICIALANALYSIS_API_KEY")
    refresh_cache = os.getenv("AA_REFRESH") == "1"
    cache_ttl_seconds = int(os.getenv("AA_CACHE_TTL_SECONDS", str(DEFAULT_CACHE_TTL_SECONDS)))

    if refresh_cache:
        cache_payload = _fetch_and_cache_models(api_key)
    else:
        try:
            cache_payload = _load_cache()
            age_seconds = int(datetime.now(UTC).timestamp()) - int(cache_payload["fetched_at_epoch_seconds"])
            if age_seconds > cache_ttl_seconds:
                cache_payload = _fetch_and_cache_models(api_key)
        except Exception:
            cache_payload = _fetch_and_cache_models(api_key)

    cutoff_date = (datetime.now(UTC) - timedelta(days=LOOKBACK_DAYS)).date().isoformat()
    enriched_models_by_row_id: dict[int, dict[str, Any]] = {}
    scoring_rows: list[dict[str, float | int | None]] = []

    for row_id, model in enumerate(cache_payload.get("models", [])):
        release_date = str(model.get("release_date") or "")
        pricing = model.get("pricing") or {}
        evaluations = model.get("evaluations") or {}

        blended_price = _as_finite_float(pricing.get("price_1m_blended_3_to_1"))
        input_price = _as_finite_float(pricing.get("price_1m_input_tokens"))
        output_price = _as_finite_float(pricing.get("price_1m_output_tokens"))
        ttfa = _as_finite_float(model.get("median_time_to_first_answer_token"))
        tps = _as_finite_float(model.get("median_output_tokens_per_second"))

        if release_date < cutoff_date:
            continue
        if not all(value is not None and value > 0 for value in [blended_price, input_price, output_price, ttfa, tps]):
            continue

        intelligence_index = _as_finite_float(evaluations.get("artificial_analysis_intelligence_index"))
        coding_index = _as_finite_float(evaluations.get("artificial_analysis_coding_index"))

        intelligence_score = None
        if intelligence_index is not None and coding_index is not None:
            intelligence_score = float((2.0 * intelligence_index) + coding_index)

        benchmark_bias_score = _mean_positive([_as_finite_float(evaluations.get(key)) for key in BENCHMARK_BIAS_KEYS])

        price_score = _inverse_log1p(blended_price)
        speed_score = float(_inverse_log1p(ttfa) + math.log(tps))
        overall_score = _mean([intelligence_score, benchmark_bias_score, price_score, speed_score])

        enriched_model = {
            **model,
            "scores": {
                "overall_score": overall_score,
                "intelligence_score": intelligence_score,
                "benchmark_bias_score": benchmark_bias_score,
                "price_score": price_score,
                "speed_score": speed_score,
            },
            "percentiles": {
                "overall_percentile": None,
                "intelligence_percentile": None,
                "speed_percentile": None,
                "price_percentile": None,
            },
        }
        enriched_models_by_row_id[row_id] = enriched_model
        scoring_rows.append(
            {
                "row_id": row_id,
                "overall_score": overall_score,
                "intelligence_score": intelligence_score,
                "speed_score": speed_score,
                "price_score": price_score,
            }
        )

    if scoring_rows:
        df = pl.DataFrame(scoring_rows).drop_nulls("overall_score")
        if df.height > 0:
            df = df.with_columns(
                [
                    (pl.col(score_column).rank("max") / pl.col(score_column).drop_nulls().count() * 100.0).alias(percentile_column)
                    for score_column, percentile_column in PERCENTILE_COLUMNS
                ]
            ).sort("overall_score", descending=True)
            ranked_models: list[dict[str, Any]] = []
            for row in df.iter_rows(named=True):
                model = enriched_models_by_row_id[int(row["row_id"])]
                model["percentiles"] = {
                    "overall_percentile": row["overall_percentile"],
                    "intelligence_percentile": row["intelligence_percentile"],
                    "speed_percentile": row["speed_percentile"],
                    "price_percentile": row["price_percentile"],
                }
                ranked_models.append(model)
            enriched_models = ranked_models
        else:
            enriched_models = []
    else:
        enriched_models = []

    output_payload = {
        "fetched_at_epoch_seconds": cache_payload["fetched_at_epoch_seconds"],
        "status_code": cache_payload["status_code"],
        "models": enriched_models,
    }
    _write_json(OUTPUT_PATH, output_payload)
    return output_payload
