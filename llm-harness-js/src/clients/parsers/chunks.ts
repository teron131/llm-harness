import { getHarnessEventStream } from "./events.js";

export async function* getStreamGenerator(
  stream: AsyncIterable<Record<string, unknown>>,
  includeReasoning = false,
): AsyncGenerator<string | [string | null, string | null]> {
  let reasoningYielded = false;

  for await (const event of getHarnessEventStream(stream, {
    includeReasoning,
    dedupeReasoning: true,
    emitReasoningBoundaries: false,
  })) {
    if (
      event.type === "reasoning-delta" &&
      includeReasoning &&
      !reasoningYielded
    ) {
      reasoningYielded = true;
      yield [event.delta, null];
      continue;
    }

    if (event.type === "text-delta") {
      yield includeReasoning ? [null, event.delta] : event.delta;
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
    const [reasoningChunk, answerChunk] = Array.isArray(item)
      ? item
      : ([null, item] as const);

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
  }

  const answer = answerParts.join("");
  return includeReasoning ? [reasoning, answer] : answer;
}
