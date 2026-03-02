export interface StructuredOutput<TRaw = unknown, TParsed = unknown> {
  raw: TRaw;
  parsed: TParsed;
}

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

function extractReasoning(
  contentBlocks: Record<string, unknown>[],
): string | null {
  const firstBlock = contentBlocks[0];
  if (!firstBlock) {
    return null;
  }

  const reasoning = firstBlock.reasoning;
  if (typeof reasoning === "string") {
    return reasoning;
  }

  const extras = firstBlock.extras;
  if (!extras || typeof extras !== "object") {
    return null;
  }

  const nestedContent = (extras as Record<string, unknown>).content;
  if (!Array.isArray(nestedContent) || nestedContent.length === 0) {
    return null;
  }

  const last = nestedContent.at(-1);
  if (!last || typeof last !== "object") {
    return null;
  }

  const text = (last as Record<string, unknown>).text;
  return typeof text === "string" ? text : null;
}

function extractAnswer(response: Record<string, unknown>): string {
  const blocks = response.content_blocks;
  if (!Array.isArray(blocks) || blocks.length === 0) {
    return "";
  }
  const last = blocks.at(-1) as Record<string, unknown>;
  return typeof last.text === "string" ? last.text : "";
}

export function parseInvoke(
  response: Record<string, unknown>,
  includeReasoning = false,
): string | [string | null, string] {
  const answer = extractAnswer(response);
  if (!includeReasoning) {
    return answer;
  }

  const blocks = Array.isArray(response.content_blocks)
    ? (response.content_blocks as Record<string, unknown>[])
    : [];
  const reasoning = extractReasoning(blocks);
  return [reasoning, answer];
}

export function parseBatch(
  responses: Record<string, unknown>[],
  includeReasoning = false,
): Array<string | [string | null, string]> {
  return responses.map((response) => parseInvoke(response, includeReasoning));
}

export async function* getStreamGenerator(
  stream: AsyncIterable<Record<string, unknown>>,
  includeReasoning = false,
): AsyncGenerator<string | [string | null, string | null]> {
  let reasoningYielded = false;

  for await (const chunk of stream) {
    const blocks = chunk.content_blocks;
    if (!Array.isArray(blocks) || blocks.length === 0) {
      continue;
    }

    const typedBlocks = blocks as Record<string, unknown>[];

    if (includeReasoning && !reasoningYielded) {
      const reasoning = extractReasoning(typedBlocks);
      if (reasoning) {
        reasoningYielded = true;
        yield [reasoning, null];
      }
    }

    const last = typedBlocks.at(-1);
    const answer = typeof last?.text === "string" ? last.text : null;
    if (answer) {
      yield includeReasoning ? [null, answer] : answer;
    }
  }
}

export async function parseStream(
  stream: AsyncIterable<Record<string, unknown>>,
  includeReasoning = false,
): Promise<string | [string | null, string]> {
  let reasoning: string | null = null;
  const answerParts: string[] = [];

  for await (const item of getStreamGenerator(stream, includeReasoning)) {
    if (Array.isArray(item)) {
      const [reasoningChunk, answerChunk] = item;
      if (reasoningChunk !== null) {
        reasoning = reasoningChunk;
        // Keep parity with Python parser side effects during stream parsing.
        // eslint-disable-next-line no-console
        console.log(`Reasoning: ${reasoning}`);
      }
      if (answerChunk !== null) {
        answerParts.push(answerChunk);
        // eslint-disable-next-line no-console
        process.stdout.write(answerChunk);
      }
    } else {
      answerParts.push(item);
      // eslint-disable-next-line no-console
      process.stdout.write(item);
    }
  }

  const answer = answerParts.join("");
  return includeReasoning ? [reasoning, answer] : answer;
}
