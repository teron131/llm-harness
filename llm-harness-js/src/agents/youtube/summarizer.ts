import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";

import { ChatOpenRouter } from "../../clients/openrouter.js";
import {
  filterContent,
  tagContent,
  untagContent,
} from "../../tools/fs/fastCopy.js";
import { getTranscript } from "../../tools/youtube/scraper.js";
import { isYoutubeUrl } from "../../utils/youtubeUtils.js";
import {
  getGarbageFilterPrompt,
  getLangchainSummaryPrompt,
  getQualityCheckPrompt,
} from "./prompts.js";
import {
  GarbageIdentificationSchema,
  QualitySchema,
  Summary,
  SummarySchema,
  isAcceptable,
  percentageScore,
} from "./schemas.js";

const SUMMARY_MODEL = "x-ai/grok-4.1-fast";
const QUALITY_MODEL = "x-ai/grok-4.1-fast";
const FAST_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025";
const MIN_QUALITY_SCORE = 80;
const MAX_ITERATIONS = 2;
const TARGET_LANGUAGE = "en";

const SummarizerState = Annotation.Root({
  transcript: Annotation<string | null>,
  summary: Annotation<Summary | null>,
  quality: Annotation<ReturnType<typeof QualitySchema.parse> | null>,
  targetLanguage: Annotation<string | null>,
  iterationCount: Annotation<number>,
  isComplete: Annotation<boolean>,
});

type SummarizerStateType = typeof SummarizerState.State;

async function garbageFilterNode(state: SummarizerStateType) {
  if (!state.transcript) {
    return {};
  }

  const taggedTranscript = tagContent(state.transcript);
  const llm = ChatOpenRouter({
    model: FAST_MODEL,
    temperature: 0,
    reasoningEffort: "low",
  }).withStructuredOutput(GarbageIdentificationSchema);

  const garbage = await llm.invoke([
    new SystemMessage(getGarbageFilterPrompt()),
    new HumanMessage(taggedTranscript),
  ]);
  if (garbage.garbage_ranges.length === 0) {
    return {};
  }

  const filtered = filterContent(taggedTranscript, garbage.garbage_ranges);
  return { transcript: untagContent(filtered) };
}

async function summaryNode(state: SummarizerStateType) {
  const llm = ChatOpenRouter({
    model: SUMMARY_MODEL,
    temperature: 0,
    reasoningEffort: "medium",
  }).withStructuredOutput(SummarySchema);

  const summary = await llm.invoke([
    new SystemMessage(
      getLangchainSummaryPrompt(state.targetLanguage ?? TARGET_LANGUAGE),
    ),
    new HumanMessage(`Transcript:\n${state.transcript ?? ""}`),
  ]);

  return {
    summary,
    iterationCount: (state.iterationCount ?? 0) + 1,
  };
}

async function qualityNode(state: SummarizerStateType) {
  const llm = ChatOpenRouter({
    model: QUALITY_MODEL,
    temperature: 0,
    reasoningEffort: "low",
  }).withStructuredOutput(QualitySchema);

  const quality = await llm.invoke([
    new SystemMessage(
      getQualityCheckPrompt(state.targetLanguage ?? TARGET_LANGUAGE),
    ),
    new HumanMessage(
      `Transcript:\n${state.transcript ?? ""}\n\nSummary:\n${JSON.stringify(state.summary ?? {})}`,
    ),
  ]);

  return {
    quality,
    isComplete: isAcceptable(quality),
  };
}

function shouldContinue(state: SummarizerStateType) {
  const qualityPercent = state.quality ? percentageScore(state.quality) : null;

  if (state.isComplete) {
    return END;
  }

  if (
    state.quality &&
    !isAcceptable(state.quality) &&
    (state.iterationCount ?? 0) < MAX_ITERATIONS
  ) {
    return "summary";
  }

  if (qualityPercent !== null && qualityPercent >= MIN_QUALITY_SCORE) {
    return END;
  }

  return END;
}

function createGraph() {
  return new StateGraph(SummarizerState)
    .addNode("garbage_filter", garbageFilterNode)
    .addNode("summary", summaryNode)
    .addNode("quality", qualityNode)
    .addEdge(START, "garbage_filter")
    .addEdge("garbage_filter", "summary")
    .addEdge("summary", "quality")
    .addConditionalEdges("quality", shouldContinue, {
      summary: "summary",
      [END]: END,
    })
    .compile();
}

async function extractTranscript(transcriptOrUrl: string): Promise<string> {
  if (isYoutubeUrl(transcriptOrUrl)) {
    return getTranscript(transcriptOrUrl);
  }
  if (!transcriptOrUrl.trim()) {
    throw new Error("Transcript cannot be empty");
  }
  return transcriptOrUrl;
}

export async function summarizeVideo({
  transcriptOrUrl,
  targetLanguage,
}: {
  transcriptOrUrl: string;
  targetLanguage?: string | null;
}): Promise<Summary> {
  const graph = createGraph();
  const transcript = await extractTranscript(transcriptOrUrl);

  const result = await graph.invoke({
    transcript,
    summary: null,
    quality: null,
    targetLanguage: targetLanguage ?? TARGET_LANGUAGE,
    iterationCount: 0,
    isComplete: false,
  });

  if (!result.summary) {
    throw new Error("Summary was not generated");
  }

  return result.summary;
}

export async function* streamSummarizeVideo({
  transcriptOrUrl,
  targetLanguage,
}: {
  transcriptOrUrl: string;
  targetLanguage?: string | null;
}) {
  const graph = createGraph();
  const transcript = await extractTranscript(transcriptOrUrl);

  const stream = await graph.stream(
    {
      transcript,
      summary: null,
      quality: null,
      targetLanguage: targetLanguage ?? TARGET_LANGUAGE,
      iterationCount: 0,
      isComplete: false,
    },
    { streamMode: "values" },
  );

  for await (const chunk of stream) {
    yield chunk;
  }
}
