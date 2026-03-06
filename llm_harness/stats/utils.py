"""Shared stats helpers aligned with the JS `src/stats/utils.ts` module."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any

import httpx

JsonObject = dict[str, Any]


def now_epoch_seconds() -> int:
    return int(datetime.now(UTC).timestamp())


def as_record(value: Any) -> JsonObject:
    return value if isinstance(value, dict) else {}


def as_finite_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if math.isfinite(numeric_value) else None


def finite_numbers(values: list[Any]) -> list[float]:
    numeric_values = [as_finite_number(value) for value in values]
    return [value for value in numeric_values if value is not None]


def mean_or_none(values: list[Any]) -> float | None:
    numeric_values = finite_numbers(values)
    if not numeric_values:
        return None
    mean_value = sum(numeric_values) / len(numeric_values)
    return round(mean_value, 4)


def percentile_rank(values: list[Any], value: Any) -> float | None:
    numeric_value = as_finite_number(value)
    if numeric_value is None:
        return None
    numeric_values = finite_numbers(values)
    if not numeric_values:
        return None
    less_or_equal_count = sum(1 for item in numeric_values if item <= numeric_value)
    raw_percentile = (less_or_equal_count / len(numeric_values)) * 100
    return round(raw_percentile, 4)


def fetch_with_timeout(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float,
    follow_redirects: bool = True,
) -> httpx.Response:
    with httpx.Client(
        timeout=timeout_seconds,
        follow_redirects=follow_redirects,
        headers=headers,
    ) as client:
        return client.get(url)


def is_fresh_epoch_seconds(fetched_at_epoch_seconds: Any, ttl_seconds: int) -> bool:
    if not isinstance(fetched_at_epoch_seconds, (int, float)):
        return False
    age_seconds = now_epoch_seconds() - int(fetched_at_epoch_seconds)
    return 0 <= age_seconds <= ttl_seconds


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
