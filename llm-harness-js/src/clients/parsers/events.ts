import {
  asRecordArray,
  extractAnswerDeltaFromChunk,
  extractReasoningDeltaFromChunk,
} from "./extractors.js";
import type {
  HarnessEventStreamOptions,
  HarnessStreamEvent,
  HarnessToolState,
} from "./types.js";

type StreamParseState = {
  reasoningOpen: boolean;
  reasoningTail: string;
  lastRawReasoningDelta: string;
};

const MAX_REASONING_TAIL_CHARS = 600;
const MAX_OVERLAP_CHARS = 300;

function calculatePrefixOverlap(
  previousText: string,
  nextDelta: string,
): number {
  const maxOverlap = Math.min(
    previousText.length,
    nextDelta.length,
    MAX_OVERLAP_CHARS,
  );

  for (let overlap = maxOverlap; overlap > 0; overlap--) {
    if (
      previousText.slice(previousText.length - overlap) ===
      nextDelta.slice(0, overlap)
    ) {
      return overlap;
    }
  }

  return 0;
}

function dedupeReasoningDelta(
  state: StreamParseState,
  rawReasoningDelta: string,
): string {
  if (state.lastRawReasoningDelta === rawReasoningDelta) {
    return "";
  }

  const overlap = calculatePrefixOverlap(
    state.reasoningTail,
    rawReasoningDelta,
  );
  const dedupedDelta = rawReasoningDelta.slice(overlap);
  state.lastRawReasoningDelta = rawReasoningDelta;

  if (!dedupedDelta) {
    return "";
  }

  state.reasoningTail = `${state.reasoningTail}${dedupedDelta}`.slice(
    -MAX_REASONING_TAIL_CHARS,
  );
  return dedupedDelta;
}

function normalizeToolState(rawType: string): HarnessToolState {
  if (rawType.includes("approval")) {
    return "approval-requested";
  }
  if (rawType.includes("error")) {
    return "output-error";
  }
  if (rawType.includes("deny")) {
    return "output-denied";
  }
  if (rawType.includes("result") || rawType.includes("output")) {
    return "output-available";
  }
  if (rawType.includes("chunk") || rawType.includes("stream")) {
    return "input-streaming";
  }
  return "input-available";
}

function extractToolEventsFromChunk(
  chunk: Record<string, unknown>,
): HarnessStreamEvent[] {
  const events: HarnessStreamEvent[] = [];
  const candidates = [
    ...asRecordArray(chunk.content_blocks),
    ...asRecordArray(chunk.tool_calls),
  ];

  for (const candidate of candidates) {
    const rawType =
      typeof candidate.type === "string" ? candidate.type.toLowerCase() : "";
    const looksLikeTool =
      rawType.includes("tool") ||
      typeof candidate.tool_call_id === "string" ||
      typeof candidate.toolCallId === "string";

    if (!looksLikeTool) {
      continue;
    }

    const toolCallId = getToolCallId(candidate);
    if (toolCallId === null) {
      continue;
    }
    events.push(buildToolEvent(candidate, rawType, toolCallId));
  }

  return events;
}

function getToolCallId(candidate: Record<string, unknown>): string | null {
  const toolCallIdValue =
    candidate.tool_call_id ??
    candidate.toolCallId ??
    candidate.id ??
    candidate.call_id;

  if (typeof toolCallIdValue !== "string" || toolCallIdValue.length === 0) {
    return null;
  }

  return toolCallIdValue;
}

function buildToolEvent(
  candidate: Record<string, unknown>,
  rawType: string,
  toolCallId: string,
): HarnessStreamEvent {
  const toolNameValue =
    candidate.name ?? candidate.tool_name ?? candidate.toolName;
  const toolName =
    typeof toolNameValue === "string" && toolNameValue.length > 0
      ? toolNameValue
      : null;

  const input = candidate.input ?? candidate.args ?? candidate.arguments;
  const output = candidate.output ?? candidate.result;
  const errorText =
    typeof candidate.error === "string"
      ? candidate.error
      : typeof candidate.errorText === "string"
        ? candidate.errorText
        : undefined;

  const event: HarnessStreamEvent = {
    type: "tool-event",
    state: normalizeToolState(rawType),
    toolCallId,
    toolName,
  };
  if (input !== undefined) {
    event.input = input;
  }
  if (output !== undefined) {
    event.output = output;
  }
  if (errorText !== undefined) {
    event.errorText = errorText;
  }

  return event;
}

export async function* getHarnessEventStream(
  stream: AsyncIterable<Record<string, unknown>>,
  options: HarnessEventStreamOptions = {},
): AsyncGenerator<HarnessStreamEvent> {
  const {
    includeReasoning = true,
    dedupeReasoning = true,
    emitReasoningBoundaries = true,
  } = options;

  const state: StreamParseState = {
    reasoningOpen: false,
    reasoningTail: "",
    lastRawReasoningDelta: "",
  };

  for await (const chunk of stream) {
    if (includeReasoning) {
      const rawReasoningDelta = extractReasoningDeltaFromChunk(chunk);
      if (
        typeof rawReasoningDelta === "string" &&
        rawReasoningDelta.trim().length > 0
      ) {
        const reasoningDelta = dedupeReasoning
          ? dedupeReasoningDelta(state, rawReasoningDelta)
          : rawReasoningDelta;

        if (reasoningDelta) {
          if (emitReasoningBoundaries && !state.reasoningOpen) {
            yield { type: "reasoning-start" };
            state.reasoningOpen = true;
          }

          yield {
            type: "reasoning-delta",
            delta: reasoningDelta,
            rawDelta: rawReasoningDelta,
          };
        }
      }
    }

    for (const toolEvent of extractToolEventsFromChunk(chunk)) {
      yield toolEvent;
    }

    const textDelta = extractAnswerDeltaFromChunk(chunk);
    if (textDelta) {
      yield { type: "text-delta", delta: textDelta };
    }
  }

  if (emitReasoningBoundaries && state.reasoningOpen) {
    yield { type: "reasoning-end" };
  }
}
