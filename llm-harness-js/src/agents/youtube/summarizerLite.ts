import { HumanMessage, SystemMessage } from "@langchain/core/messages";

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
} from "./prompts.js";
import {
  GarbageIdentificationSchema,
  Summary,
  SummarySchema,
} from "./schemas.js";

const DEFAULT_MODEL = "google/gemini-3-flash-preview";
const FAST_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025";

async function scrapeYoutube(youtubeUrl: string): Promise<string> {
  return getTranscript(youtubeUrl);
}

async function garbageFilterTranscript(transcript: string): Promise<string> {
  const taggedTranscript = tagContent(transcript);

  const llm = ChatOpenRouter({
    model: FAST_MODEL,
    temperature: 0,
  }).withStructuredOutput(GarbageIdentificationSchema);
  const garbage = await llm.invoke([
    new SystemMessage(getGarbageFilterPrompt()),
    new HumanMessage(taggedTranscript),
  ]);

  if (!garbage.garbage_ranges.length) {
    return transcript;
  }

  const filtered = filterContent(taggedTranscript, garbage.garbage_ranges);
  return untagContent(filtered);
}

export async function summarizeVideo({
  transcriptOrUrl,
  targetLanguage,
}: {
  transcriptOrUrl: string;
  targetLanguage?: string | null;
}): Promise<Summary> {
  const llm = ChatOpenRouter({
    model: DEFAULT_MODEL,
    temperature: 0,
    reasoningEffort: "medium",
  }).withStructuredOutput(SummarySchema);

  const transcript = isYoutubeUrl(transcriptOrUrl)
    ? await scrapeYoutube(transcriptOrUrl)
    : transcriptOrUrl;
  const cleanedTranscript = await garbageFilterTranscript(transcript);

  const summary = await llm.invoke([
    new SystemMessage(getLangchainSummaryPrompt(targetLanguage ?? null)),
    new HumanMessage(`Transcript:\n${cleanedTranscript}`),
  ]);

  return summary;
}
