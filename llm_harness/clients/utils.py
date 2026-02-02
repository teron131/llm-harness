from collections.abc import Generator

from langchain_core.messages import AIMessage


def _extract_reasoning(content_blocks: list[dict]) -> str | None:
    """Extract reasoning from response content_blocks."""
    if not content_blocks:
        return None

    block = content_blocks[0]
    if reasoning := block.get("reasoning"):
        return reasoning

    if (
        (extras := block.get("extras"))
        and isinstance(extras, dict)
        and (content := extras.get("content"))
        and isinstance(content, list)
        and content
        and isinstance(content[-1], dict)
    ):
        return content[-1].get("text")

    return None


def parse_invoke(
    response: AIMessage,
    include_reasoning: bool = False,
) -> str | tuple[str | None, str]:
    """Parse response to extract answer and optionally reasoning."""
    answer = response.content_blocks[-1]["text"]
    if include_reasoning:
        reasoning = _extract_reasoning(response.content_blocks)
        return reasoning, answer
    return answer


def parse_batch(
    responses: list[AIMessage],
    include_reasoning: bool = False,
) -> list[str] | list[tuple[str | None, str]]:
    """Parse batched responses, optionally with reasoning."""
    return [parse_invoke(response, include_reasoning) for response in responses]


def get_stream_generator(
    stream: Generator[AIMessage, None, None],
    include_reasoning: bool = False,
) -> Generator[str | tuple[str, str | None], None, None]:
    """Yield streaming chunks, optionally with reasoning."""
    reasoning_yielded = False

    for chunk in stream:
        if not (blocks := getattr(chunk, "content_blocks", None)):
            continue

        if include_reasoning and not reasoning_yielded and (reasoning := _extract_reasoning(blocks)):
            reasoning_yielded = True
            yield (reasoning, None)

        if answer := blocks[-1].get("text"):
            yield (None, answer) if include_reasoning else answer


def parse_stream(
    stream: Generator[AIMessage, None, None],
    include_reasoning: bool = False,
) -> str | tuple[str | None, str]:
    """Print streamed chunks and return the final result."""
    reasoning = None
    answer_parts: list[str] = []

    for item in get_stream_generator(stream, include_reasoning):
        if isinstance(item, tuple):
            reasoning_chunk, answer_chunk = item
            if reasoning_chunk is not None:
                reasoning = reasoning_chunk
                print(f"Reasoning: {reasoning}", flush=True)
            if answer_chunk is not None:
                answer_parts.append(answer_chunk)
                print(answer_chunk, end="", flush=True)
        else:
            answer_parts.append(item)
            print(item, end="", flush=True)

    answer = "".join(answer_parts)
    return (reasoning, answer) if include_reasoning else answer
