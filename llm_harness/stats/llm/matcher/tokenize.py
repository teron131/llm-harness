"""Tokenization helpers for native Python LLM matcher scoring."""

from __future__ import annotations

import re

from ..shared import normalize_model_token

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


def split_base_model_id(model_id: str) -> str:
    return model_id.split("/")[-1] if "/" in model_id else model_id


def _is_b_scale_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d+b", token) or re.fullmatch(r"a\d+b", token))


def _split_mixed_alphanumeric_token(token: str) -> list[str]:
    if _is_b_scale_token(token):
        return [token]
    return [part for part in re.split(r"(?<=\D)(?=\d)|(?<=\d)(?=\D)", token) if part]


def split_tokens(value: str) -> list[str]:
    return [token for chunk in normalize_model_token(value).split("-") for token in _split_mixed_alphanumeric_token(chunk) if token and token not in MODEL_NAME_TAG_TOKENS]


def split_base_model_tokens(model_id: str) -> list[str]:
    return split_tokens(split_base_model_id(model_id))


def first_parsed_number(tokens: list[str], parser) -> int | None:
    for token in tokens:
        parsed = parser(token)
        if parsed is not None:
            return parsed
    return None


def is_numeric_token(token: str | None) -> bool:
    return bool(token and re.fullmatch(r"\d+", token))


def parse_numeric_or_b_scale_token(token: str | None) -> int | None:
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


def parse_b_scale_token(token: str | None) -> int | None:
    if not token:
        return None
    match = re.fullmatch(r"(\d+)b", token)
    return int(match.group(1)) if match else None


def parse_active_b_token(token: str | None) -> int | None:
    if not token:
        return None
    match = re.fullmatch(r"a(\d+)b", token)
    return int(match.group(1)) if match else None


def parsed_numeric_tokens(tokens: list[str]) -> list[int]:
    return [value for value in (parse_numeric_or_b_scale_token(token) for token in tokens) if value is not None]


def common_prefix_length(left: str, right: str) -> int:
    max_length = min(len(left), len(right))
    index = 0
    while index < max_length and left[index] == right[index]:
        index += 1
    return index
