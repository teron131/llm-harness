"""Cross-source model matching and unioning (Artificial Analysis + models.dev)."""

from __future__ import annotations

import re
from typing import Any, TypedDict

from .artificial_analysis import get_artificial_analysis_stats
from .models_dev import get_models_dev_stats

TOKEN_PREFIX_WEIGHTS = (5, 4, 3, 2, 1)
DEFAULT_MAX_CANDIDATES = 5
TOKEN_PREFIX_REWARD_MULTIPLIER = 2
NUMERIC_EXACT_MATCH_REWARD = 2
NUMERIC_CLOSENESS_REWARD_SCALE = 0.1
NUMERIC_ALL_EQUAL_REWARD = 0.2
VARIANT_SUFFIX_REWARD = 2
COVERAGE_EXACT_REWARD = 4
COVERAGE_MISSING_BASE_PENALTY = 1
B_SCALE_EXACT_REWARD = 3
B_SCALE_MISMATCH_PENALTY = 4
B_SCALE_MISSING_PENALTY = 2
ACTIVE_B_EXACT_REWARD = 2
ACTIVE_B_MISMATCH_PENALTY = 2
CHAR_PREFIX_REWARD_SCALE = 0.03
LENGTH_GAP_PENALTY_SCALE = 0.005
PROVIDER_FILTER = "openrouter"
VOID_THRESHOLD_RANGE_RATIO = 0.35
MODEL_NAME_TAG_TOKENS = {
    "free",
    "extended",
    "exacto",
    "instruct",
    "vl",
    "thinking",
    "reasoning",
    "online",
    "nitro",
}


class MatchCandidate(TypedDict):
    """models.dev candidate for one Artificial Analysis model."""

    model_id: str
    provider_id: str
    provider_name: str
    model_name: str | None
    score: float


MatchResult = MatchCandidate | None


class MatchMappedModel(TypedDict):
    """Mapping entry for one Artificial Analysis model."""

    artificial_analysis_slug: str
    artificial_analysis_name: str | None
    artificial_analysis_release_date: str | None
    best_match: MatchResult
    candidates: list[MatchCandidate]


class MatchModelMappingOptions(TypedDict, total=False):
    """Options for mapping generation."""

    max_candidates: int


class MatchModelsUnionOptions(TypedDict, total=False):
    """Options for union generation."""

    max_candidates: int


class MatchModelMappingPayload(TypedDict):
    """Full mapping payload."""

    artificial_analysis_fetched_at_epoch_seconds: int | None
    models_dev_fetched_at_epoch_seconds: int | None
    total_artificial_analysis_models: int
    total_models_dev_models: int
    max_candidates: int
    void_mode: str
    void_threshold: float | None
    voided_count: int
    models: list[MatchMappedModel]


class MatchModelsUnionPayload(TypedDict):
    """Union payload from matched models."""

    artificial_analysis_fetched_at_epoch_seconds: int | None
    models_dev_fetched_at_epoch_seconds: int | None
    total_artificial_analysis_models: int
    total_models_dev_models: int
    void_mode: str
    void_threshold: float | None
    voided_count: int
    total_union_models: int
    models: list[dict[str, Any]]


def _normalize(value: str) -> str:
    normalized = value.lower()
    normalized = re.sub(r"[._:\s]+", "-", normalized)
    normalized = re.sub(r"[^a-z0-9/-]+", "", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-/")


def _split_base_model_id(model_id: str) -> str:
    parts = model_id.split("/")
    return parts[-1] if parts else model_id


def _is_b_scale_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d+b", token) or re.fullmatch(r"a\d+b", token))


def _split_mixed_alphanumeric_token(token: str) -> list[str]:
    if _is_b_scale_token(token):
        return [token]
    return [part for part in re.split(r"(?<=\D)(?=\d)|(?<=\d)(?=\D)", token) if part]


def _split_tokens(value: str) -> list[str]:
    tokens = []
    for token in _normalize(value).split("-"):
        for part in _split_mixed_alphanumeric_token(token):
            if part and part not in MODEL_NAME_TAG_TOKENS:
                tokens.append(part)
    return tokens


def _first_parsed_number(tokens: list[str], parser: Any) -> int | None:
    for token in tokens:
        parsed_value = parser(token)
        if parsed_value is not None:
            return parsed_value
    return None


def _is_numeric_token(token: str | None) -> bool:
    return bool(token and re.fullmatch(r"\d+", token))


def _parse_numeric_or_b_scale_token(token: str | None) -> int | None:
    if not token:
        return None
    if re.fullmatch(r"\d+", token):
        return int(token)
    match = re.fullmatch(r"(\d+)b", token)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"a(\d+)b", token)
    if match:
        return int(match.group(1))
    return None


def _parse_b_scale_token(token: str | None) -> int | None:
    if not token:
        return None
    match = re.fullmatch(r"(\d+)b", token)
    return int(match.group(1)) if match else None


def _parse_active_b_token(token: str | None) -> int | None:
    if not token:
        return None
    match = re.fullmatch(r"a(\d+)b", token)
    return int(match.group(1)) if match else None


def _common_prefix_length(left: str, right: str) -> int:
    max_len = min(len(left), len(right))
    idx = 0
    while idx < max_len and left[idx] == right[idx]:
        idx += 1
    return idx


def _weighted_token_prefix_score(left_tokens: list[str], right_tokens: list[str]) -> int:
    max_len = min(len(left_tokens), len(right_tokens))
    score = 0
    for idx in range(max_len):
        if left_tokens[idx] != right_tokens[idx]:
            break
        score += TOKEN_PREFIX_WEIGHTS[idx] if idx < len(TOKEN_PREFIX_WEIGHTS) else 0
    return score


def _numeric_match_reward(aa_slug: str, model_id: str) -> float:
    aa_tokens = _split_tokens(aa_slug)
    model_tokens = _split_tokens(_split_base_model_id(model_id))
    max_len = min(len(aa_tokens), len(model_tokens))
    for idx in range(max_len):
        aa_numeric_value = _parse_numeric_or_b_scale_token(aa_tokens[idx])
        model_numeric_value = _parse_numeric_or_b_scale_token(model_tokens[idx])
        if aa_numeric_value is not None and model_numeric_value is not None:
            return float(NUMERIC_EXACT_MATCH_REWARD if aa_numeric_value == model_numeric_value else 0)
    return 0.0


def _numeric_closeness_reward(aa_slug: str, model_id: str) -> float:
    aa_numbers = [value for value in (_parse_numeric_or_b_scale_token(token) for token in _split_tokens(aa_slug)) if value is not None]
    model_numbers = [value for value in (_parse_numeric_or_b_scale_token(token) for token in _split_tokens(_split_base_model_id(model_id))) if value is not None]
    max_len = max(len(aa_numbers), len(model_numbers))
    for idx in range(max_len):
        aa_value = aa_numbers[idx] if idx < len(aa_numbers) else None
        model_value = model_numbers[idx] if idx < len(model_numbers) else None
        if aa_value is None or model_value is None:
            return 0.0
        if aa_value == model_value:
            continue
        return NUMERIC_CLOSENESS_REWARD_SCALE / (1 + abs(aa_value - model_value))
    return NUMERIC_ALL_EQUAL_REWARD


def _b_scale_reward_or_penalty(aa_slug: str, model_id: str, model_name: str) -> float:
    aa_tokens = _split_tokens(aa_slug)
    model_base_tokens = _split_tokens(_split_base_model_id(model_id))
    model_name_tokens = _split_tokens(model_name)
    aa_b_scale = _first_parsed_number(aa_tokens, _parse_b_scale_token)
    if aa_b_scale is None:
        return 0.0
    base_b_scale = _first_parsed_number(model_base_tokens, _parse_b_scale_token)
    name_b_scale = _first_parsed_number(model_name_tokens, _parse_b_scale_token)
    candidate_b_scale = base_b_scale if base_b_scale is not None else name_b_scale
    if candidate_b_scale is None:
        return float(-B_SCALE_MISSING_PENALTY)
    if candidate_b_scale == aa_b_scale:
        return float(B_SCALE_EXACT_REWARD)
    return float(-B_SCALE_MISMATCH_PENALTY)


def _has_hard_b_scale_mismatch(aa_slug: str, model_id: str, model_name: str) -> bool:
    aa_b_scale = _first_parsed_number(_split_tokens(aa_slug), _parse_b_scale_token)
    if aa_b_scale is None:
        return False
    model_base_b_scale = _first_parsed_number(_split_tokens(_split_base_model_id(model_id)), _parse_b_scale_token)
    model_name_b_scale = _first_parsed_number(_split_tokens(model_name), _parse_b_scale_token)
    candidate_b_scale = model_base_b_scale if model_base_b_scale is not None else model_name_b_scale
    if candidate_b_scale is None:
        return False
    return candidate_b_scale != aa_b_scale


def _active_b_reward_or_penalty(aa_slug: str, model_id: str, model_name: str) -> float:
    aa_active_b = _first_parsed_number(_split_tokens(aa_slug), _parse_active_b_token)
    if aa_active_b is None:
        return 0.0
    base_active_b = _first_parsed_number(_split_tokens(_split_base_model_id(model_id)), _parse_active_b_token)
    name_active_b = _first_parsed_number(_split_tokens(model_name), _parse_active_b_token)
    candidate_active_b = base_active_b if base_active_b is not None else name_active_b
    if candidate_active_b is None:
        return 0.0
    if candidate_active_b == aa_active_b:
        return float(ACTIVE_B_EXACT_REWARD)
    return float(-ACTIVE_B_MISMATCH_PENALTY)


def _same_variant_reward(aa_slug: str, model_id: str, model_name: str) -> float:
    aa_tokens = _split_tokens(aa_slug)
    model_base_tokens = _split_tokens(_split_base_model_id(model_id))
    model_name_tokens = _split_tokens(model_name)
    aa_last_token = aa_tokens[-1] if aa_tokens else None
    if not aa_last_token or _is_numeric_token(aa_last_token):
        return 0.0
    base_last_token = model_base_tokens[-1] if model_base_tokens else None
    name_last_token = model_name_tokens[-1] if model_name_tokens else None
    if aa_last_token in (base_last_token, name_last_token):
        return float(VARIANT_SUFFIX_REWARD)
    return 0.0


def _coverage_reward_or_penalty(aa_slug: str, model_id: str, model_name: str) -> float:
    aa_set = set(_split_tokens(aa_slug))
    base_set = set(_split_tokens(_split_base_model_id(model_id)))
    name_set = set(_split_tokens(model_name))

    def compare_sets(candidate_set: set[str]) -> float:
        if not aa_set:
            return 0.0
        missing_count = sum(1 for token in aa_set if token not in candidate_set)
        if missing_count > 0:
            return float(-COVERAGE_MISSING_BASE_PENALTY - missing_count)
        if len(candidate_set) == len(aa_set):
            return float(COVERAGE_EXACT_REWARD)
        return 0.0

    return max(compare_sets(base_set), compare_sets(name_set))


def _has_first_token_match(aa_slug: str, model_id: str, model_name: str) -> bool:
    aa_tokens = _split_tokens(aa_slug)
    if not aa_tokens:
        return False
    aa_first = aa_tokens[0]
    base_tokens = _split_tokens(_split_base_model_id(model_id))
    name_tokens = _split_tokens(model_name)
    return (base_tokens and base_tokens[0] == aa_first) or (name_tokens and name_tokens[0] == aa_first)


def _score_candidate(aa_slug: str, model_id: str, model_name: str) -> float:
    normalized_aa = _normalize(aa_slug)
    normalized_model_base = _normalize(_split_base_model_id(model_id))
    normalized_model_name = _normalize(model_name)

    aa_tokens = _split_tokens(aa_slug)
    model_base_tokens = _split_tokens(_split_base_model_id(model_id))
    model_name_tokens = _split_tokens(model_name)
    prefix_base = _common_prefix_length(normalized_aa, normalized_model_base)
    prefix_name = _common_prefix_length(normalized_aa, normalized_model_name)
    max_prefix = max(prefix_base, prefix_name)
    if max_prefix == 0:
        return 0.0
    if _has_hard_b_scale_mismatch(aa_slug, model_id, model_name):
        return 0.0

    weighted_token_score = max(
        _weighted_token_prefix_score(aa_tokens, model_base_tokens),
        _weighted_token_prefix_score(aa_tokens, model_name_tokens),
    )
    return (
        (weighted_token_score * TOKEN_PREFIX_REWARD_MULTIPLIER)
        + _numeric_match_reward(aa_slug, model_id)
        + _numeric_closeness_reward(aa_slug, model_id)
        + _same_variant_reward(aa_slug, model_id, model_name)
        + _b_scale_reward_or_penalty(aa_slug, model_id, model_name)
        + _active_b_reward_or_penalty(aa_slug, model_id, model_name)
        + _coverage_reward_or_penalty(aa_slug, model_id, model_name)
        + (max_prefix * CHAR_PREFIX_REWARD_SCALE)
        - (abs(len(normalized_aa) - len(normalized_model_base)) * LENGTH_GAP_PENALTY_SCALE)
    )


def _scope_to_openrouter_models(models_dev_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [model for model in models_dev_models if model.get("provider_id") == PROVIDER_FILTER]


def _collect_candidates_for_aa_model(aa_model: dict[str, Any], models_dev_models: list[dict[str, Any]]) -> list[MatchCandidate]:
    aa_slug = str(aa_model.get("slug") or "")
    if not aa_slug:
        return []
    candidates: list[MatchCandidate] = []
    for model_stats_model in models_dev_models:
        model_name = model_stats_model.get("model", {}).get("name")
        model_name = model_name if isinstance(model_name, str) else ""
        model_id = str(model_stats_model.get("model_id") or "")
        if not _has_first_token_match(aa_slug, model_id, model_name):
            continue
        score = _score_candidate(aa_slug, model_id, model_name)
        if score <= 0:
            continue
        candidates.append(
            {
                "model_id": model_id,
                "provider_id": str(model_stats_model.get("provider_id") or ""),
                "provider_name": str(model_stats_model.get("provider_name") or ""),
                "model_name": model_name or None,
                "score": float(score),
            }
        )
    return sorted(candidates, key=lambda item: (-item["score"], item["model_id"]))


def _apply_maxmin_half_void(models: list[dict[str, Any]]) -> tuple[float | None, int]:
    scores = sorted(
        [
            float(model.get("best_match", {}).get("score"))
            for model in models
            if isinstance(model.get("best_match"), dict) and isinstance(model.get("best_match", {}).get("score"), (int, float))
        ]
    )
    if not scores:
        return None, 0
    min_score = scores[0]
    max_score = scores[-1]
    threshold = min_score + ((max_score - min_score) * VOID_THRESHOLD_RANGE_RATIO)
    voided = 0
    for model in models:
        best_match = model.get("best_match")
        score = best_match.get("score") if isinstance(best_match, dict) else None
        if isinstance(score, (int, float)) and score < threshold:
            model["best_match"] = None
            if isinstance(model.get("candidates"), list):
                model["candidates"] = []
            voided += 1
    return float(threshold), voided


def get_match_models_union(
    _options: MatchModelsUnionOptions | None = None,
) -> MatchModelsUnionPayload:
    """Build union rows from matched Artificial Analysis and models.dev models."""
    try:
        artificial_analysis_stats = get_artificial_analysis_stats()
        models_dev_stats = get_models_dev_stats()
        scoped_models_dev_models = _scope_to_openrouter_models(models_dev_stats.get("models") or [])

        rows: list[dict[str, Any]] = []
        for aa_model in artificial_analysis_stats.get("models") or []:
            candidates = _collect_candidates_for_aa_model(aa_model, scoped_models_dev_models)
            best_match = candidates[0] if candidates else None
            matched_models_dev = None
            if best_match is not None:
                matched_models_dev = next((model for model in scoped_models_dev_models if model.get("model_id") == best_match.get("model_id")), None)

            union_payload = {
                **(matched_models_dev.get("model") if isinstance(matched_models_dev, dict) else {}),
                **aa_model,
                "name": (((matched_models_dev.get("model") or {}).get("name") if isinstance(matched_models_dev, dict) else None) or aa_model.get("name") or None),
            }
            rows.append(
                {
                    "artificial_analysis_slug": aa_model.get("slug") if isinstance(aa_model.get("slug"), str) else "",
                    "artificial_analysis_name": aa_model.get("name") if isinstance(aa_model.get("name"), str) else None,
                    "artificial_analysis_release_date": aa_model.get("release_date") if isinstance(aa_model.get("release_date"), str) else None,
                    "best_match": best_match,
                    "artificial_analysis": aa_model,
                    "models_dev": matched_models_dev,
                    "union": union_payload,
                }
            )

        void_threshold, voided_count = _apply_maxmin_half_void(rows)
        unions = [row["union"] for row in rows if row.get("best_match") is not None]

        return {
            "artificial_analysis_fetched_at_epoch_seconds": artificial_analysis_stats.get("fetched_at_epoch_seconds"),
            "models_dev_fetched_at_epoch_seconds": models_dev_stats.get("fetched_at_epoch_seconds"),
            "total_artificial_analysis_models": len(artificial_analysis_stats.get("models") or []),
            "total_models_dev_models": len(scoped_models_dev_models),
            "void_mode": "maxmin_half",
            "void_threshold": void_threshold,
            "voided_count": voided_count,
            "total_union_models": len(unions),
            "models": unions,
        }
    except Exception:
        return {
            "artificial_analysis_fetched_at_epoch_seconds": None,
            "models_dev_fetched_at_epoch_seconds": None,
            "total_artificial_analysis_models": 0,
            "total_models_dev_models": 0,
            "void_mode": "maxmin_half",
            "void_threshold": None,
            "voided_count": 0,
            "total_union_models": 0,
            "models": [],
        }


def get_match_model_mapping(
    options: MatchModelMappingOptions | None = None,
) -> MatchModelMappingPayload:
    """Build candidate mappings from Artificial Analysis models to models.dev models."""
    options = options or {}
    max_candidates = int(options.get("max_candidates") or DEFAULT_MAX_CANDIDATES)
    try:
        artificial_analysis_stats = get_artificial_analysis_stats()
        models_dev_stats = get_models_dev_stats()
        scoped_models_dev_models = _scope_to_openrouter_models(models_dev_stats.get("models") or [])

        models: list[MatchMappedModel] = []
        for aa_model in artificial_analysis_stats.get("models") or []:
            candidates = _collect_candidates_for_aa_model(aa_model, scoped_models_dev_models)[:max_candidates]
            models.append(
                {
                    "artificial_analysis_slug": aa_model.get("slug") if isinstance(aa_model.get("slug"), str) else "",
                    "artificial_analysis_name": aa_model.get("name") if isinstance(aa_model.get("name"), str) else None,
                    "artificial_analysis_release_date": aa_model.get("release_date") if isinstance(aa_model.get("release_date"), str) else None,
                    "best_match": candidates[0] if candidates else None,
                    "candidates": candidates,
                }
            )

        models_mutable = [dict(model) for model in models]
        void_threshold, voided_count = _apply_maxmin_half_void(models_mutable)

        return {
            "artificial_analysis_fetched_at_epoch_seconds": artificial_analysis_stats.get("fetched_at_epoch_seconds"),
            "models_dev_fetched_at_epoch_seconds": models_dev_stats.get("fetched_at_epoch_seconds"),
            "total_artificial_analysis_models": len(models),
            "total_models_dev_models": len(scoped_models_dev_models),
            "max_candidates": max_candidates,
            "void_mode": "maxmin_half",
            "void_threshold": void_threshold,
            "voided_count": voided_count,
            "models": models_mutable,
        }
    except Exception:
        return {
            "artificial_analysis_fetched_at_epoch_seconds": None,
            "models_dev_fetched_at_epoch_seconds": None,
            "total_artificial_analysis_models": 0,
            "total_models_dev_models": 0,
            "max_candidates": max_candidates,
            "void_mode": "maxmin_half",
            "void_threshold": None,
            "voided_count": 0,
            "models": [],
        }
