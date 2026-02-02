"""Response parsing utilities with metadata extraction."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict


class StructuredOutput(BaseModel):
    """Parse the raw message from the structured output response.

    >>> StructuredOutput = ChatOpenAI.with_structured_output(Schema, include_raw=True)
    """

    model_config = ConfigDict(extra="ignore")
    raw: AIMessage
    parsed: BaseModel


def get_metadata(ai_message: AIMessage) -> tuple[int, int, float]:
    """Return (input_tokens, output_tokens, cost) from an AIMessage.

    - Prefer `AIMessage.usage_metadata` (LangChain standard; works for Gemini).
    - Fall back to `response_metadata["token_usage"]` (OpenAI/OpenRouter; includes cost here).
    - Fall back to legacy Gemini shapes under `response_metadata["usage_metadata"]`.

    Cost may be unavailable for some providers; returns 0.0.
    """
    um = getattr(ai_message, "usage_metadata", None)
    if isinstance(um, dict) and um:
        return int(um.get("input_tokens") or 0), int(um.get("output_tokens") or 0), 0.0

    rm = getattr(ai_message, "response_metadata", None)
    if not isinstance(rm, dict) or not rm:
        return 0, 0, 0.0

    token_usage = rm.get("token_usage")
    if isinstance(token_usage, dict) and token_usage:
        return (
            int(token_usage.get("prompt_tokens") or 0),
            int(token_usage.get("completion_tokens") or 0),
            float(token_usage.get("cost") or 0.0),
        )

    usage_md = rm.get("usage_metadata")
    if isinstance(usage_md, dict) and usage_md:
        input_tokens = int(usage_md.get("prompt_token_count") or usage_md.get("input_token_count") or 0)
        output_tokens = int(usage_md.get("candidates_token_count") or usage_md.get("output_token_count") or 0)
        return input_tokens, output_tokens, 0.0

    return 0, 0, 0.0
