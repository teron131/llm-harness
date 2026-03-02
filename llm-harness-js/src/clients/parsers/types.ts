export interface StructuredOutput<TRaw = unknown, TParsed = unknown> {
  raw: TRaw;
  parsed: TParsed;
}

export type HarnessToolState =
  | "input-streaming"
  | "input-available"
  | "approval-requested"
  | "output-available"
  | "output-error"
  | "output-denied";

export type HarnessStreamEvent =
  | { type: "reasoning-start" }
  | { type: "reasoning-delta"; delta: string; rawDelta: string }
  | { type: "reasoning-end" }
  | { type: "text-delta"; delta: string }
  | {
      type: "tool-event";
      state: HarnessToolState;
      toolCallId: string;
      toolName: string | null;
      input?: unknown;
      output?: unknown;
      errorText?: string;
    };

export interface HarnessEventStreamOptions {
  includeReasoning?: boolean;
  dedupeReasoning?: boolean;
  emitReasoningBoundaries?: boolean;
}
