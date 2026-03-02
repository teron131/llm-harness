function toInt(value: unknown): number {
  return Number(value ?? 0);
}

function toFloat(value: unknown): number {
  return Number(value ?? 0);
}

export function getMetadata(
  aiMessage: Record<string, unknown>,
): [number, number, number] {
  const usageMetadata = aiMessage.usage_metadata;
  if (usageMetadata && typeof usageMetadata === "object") {
    const typed = usageMetadata as Record<string, unknown>;
    return [toInt(typed.input_tokens), toInt(typed.output_tokens), 0];
  }

  const responseMetadata = aiMessage.response_metadata;
  if (!responseMetadata || typeof responseMetadata !== "object") {
    return [0, 0, 0];
  }

  const response = responseMetadata as Record<string, unknown>;
  const tokenUsage = response.token_usage;
  if (tokenUsage && typeof tokenUsage === "object") {
    const typed = tokenUsage as Record<string, unknown>;
    return [
      toInt(typed.prompt_tokens),
      toInt(typed.completion_tokens),
      toFloat(typed.cost),
    ];
  }

  const legacyUsage = response.usage_metadata;
  if (legacyUsage && typeof legacyUsage === "object") {
    const typed = legacyUsage as Record<string, unknown>;
    const inputTokens = toInt(
      typed.prompt_token_count ?? typed.input_token_count,
    );
    const outputTokens = toInt(
      typed.candidates_token_count ?? typed.output_token_count,
    );
    return [inputTokens, outputTokens, 0];
  }

  return [0, 0, 0];
}
