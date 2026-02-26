import {
  formatYoutubeLoaderOutput,
  scrapeYoutube,
} from "../../tools/youtube/scraper.js";
import { Summary } from "./schemas.js";

export { SummarySchema } from "./schemas.js";

export async function summarizeVideo(
  transcriptOrUrl: string,
  targetLanguage?: string | null,
): Promise<Summary> {
  const { summarizeVideo } = await import("./summarizerLite.js");
  return summarizeVideo({
    transcriptOrUrl,
    ...(targetLanguage !== undefined ? { targetLanguage } : {}),
  });
}

export async function streamSummarizeVideo(
  transcriptOrUrl: string,
  targetLanguage?: string | null,
) {
  const { streamSummarizeVideo } = await import("./summarizer.js");
  return streamSummarizeVideo({
    transcriptOrUrl,
    ...(targetLanguage !== undefined ? { targetLanguage } : {}),
  });
}

export async function summarizeVideoReact(
  transcriptOrUrl: string,
  targetLanguage?: string | null,
): Promise<Summary> {
  const { summarizeVideo } = await import("./summarizer.js");
  return summarizeVideo({
    transcriptOrUrl,
    ...(targetLanguage !== undefined ? { targetLanguage } : {}),
  });
}

export async function youtubeLoader(url: string): Promise<string> {
  const result = await scrapeYoutube(url);
  return formatYoutubeLoaderOutput(result);
}
