import { z } from "zod";

import {
  cleanText,
  cleanYoutubeUrl,
  extractVideoId,
  isYoutubeUrl,
} from "../../utils/youtubeUtils.js";

const SCRAPECREATORS_ENDPOINT =
  "https://api.scrapecreators.com/v1/youtube/video/transcript";
const SUPADATA_ENDPOINT = "https://api.supadata.ai/v1/transcript";
const DEFAULT_TIMEOUT_MS = 30_000;

function getApiKey(name: string): string | null {
  const value = process.env[name]?.trim();
  return value ? value : null;
}

export const ChannelSchema = z
  .object({
    id: z.string().optional(),
    url: z.string().optional(),
    handle: z.string().optional(),
    title: z.string().optional(),
  })
  .passthrough();

export const TranscriptSegmentSchema = z
  .object({
    text: z.string().optional(),
    startMs: z.number().optional(),
    endMs: z.number().optional(),
    startTimeText: z.string().optional(),
  })
  .passthrough();

export const YouTubeScraperResultSchema = z
  .object({
    success: z.boolean().optional(),
    credits_remaining: z.number().optional(),
    type: z.string().optional(),
    transcript: z.array(TranscriptSegmentSchema).optional(),
    transcript_only_text: z.string().optional(),
    title: z.string().optional(),
    description: z.string().optional(),
    thumbnail: z.string().optional(),
    url: z.string().optional(),
    id: z.string().optional(),
    viewCountInt: z.number().optional(),
    likeCountInt: z.number().optional(),
    publishDate: z.string().optional(),
    publishDateText: z.string().optional(),
    channel: ChannelSchema.optional(),
    durationFormatted: z.string().optional(),
    keywords: z.array(z.string()).optional(),
    videoId: z.string().optional(),
    captionTracks: z.array(z.record(z.unknown())).optional(),
    language: z.string().optional(),
    availableLangs: z.array(z.string()).optional(),
  })
  .passthrough();

export type YouTubeScraperResult = z.infer<typeof YouTubeScraperResultSchema>;

function hasTranscript(result: YouTubeScraperResult): boolean {
  return Boolean(
    result.transcript?.length || result.transcript_only_text?.trim(),
  );
}

function parsedTranscript(result: YouTubeScraperResult): string | null {
  if (result.transcript?.length) {
    return cleanText(
      result.transcript
        .map((segment) => segment.text)
        .filter(Boolean)
        .join(" "),
    );
  }
  if (result.transcript_only_text?.trim()) {
    return cleanText(result.transcript_only_text);
  }
  return null;
}

async function fetchWithTimeout(
  url: string,
  init?: RequestInit,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

async function fetchScrapeCreators(
  videoUrl: string,
): Promise<YouTubeScraperResult | null> {
  const apiKey = getApiKey("SCRAPECREATORS_API_KEY");
  if (!apiKey) {
    return null;
  }

  try {
    const url = new URL(SCRAPECREATORS_ENDPOINT);
    url.searchParams.set("url", videoUrl);

    const response = await fetchWithTimeout(url.toString(), {
      headers: { "x-api-key": apiKey },
    });

    if ([401, 403].includes(response.status) || !response.ok) {
      return null;
    }

    const data = (await response.json()) as unknown;
    return YouTubeScraperResultSchema.parse(data);
  } catch {
    return null;
  }
}

async function fetchSupadata(
  videoUrl: string,
): Promise<YouTubeScraperResult | null> {
  const apiKey = getApiKey("SUPADATA_API_KEY");
  if (!apiKey) {
    return null;
  }

  try {
    const url = new URL(SUPADATA_ENDPOINT);
    url.searchParams.set("url", videoUrl);
    url.searchParams.set("lang", "en");
    url.searchParams.set("text", "true");
    url.searchParams.set("mode", "auto");

    const response = await fetchWithTimeout(url.toString(), {
      headers: { "x-api-key": apiKey },
    });

    if ([401, 403, 202].includes(response.status) || !response.ok) {
      return null;
    }

    const data = (await response.json()) as Record<string, unknown>;
    const content = data.content;

    let transcript_only_text: string | undefined;
    let transcript: Array<z.infer<typeof TranscriptSegmentSchema>> | undefined;

    if (typeof content === "string") {
      transcript_only_text = content;
    } else if (Array.isArray(content)) {
      transcript = content
        .filter(
          (item): item is Record<string, unknown> =>
            typeof item === "object" && item !== null,
        )
        .map((item) => {
          const offset = typeof item.offset === "number" ? item.offset : 0;
          const duration =
            typeof item.duration === "number" ? item.duration : 0;
          return {
            text: typeof item.text === "string" ? item.text : undefined,
            startMs: offset,
            endMs: offset + duration,
            startTimeText: undefined,
          };
        });
    }

    return {
      url: videoUrl,
      transcript,
      transcript_only_text,
      videoId: extractVideoId(videoUrl) ?? undefined,
      language: typeof data.lang === "string" ? data.lang : undefined,
      availableLangs: Array.isArray(data.availableLangs)
        ? data.availableLangs.filter((x): x is string => typeof x === "string")
        : undefined,
      success: true,
      type: "video",
    };
  } catch {
    return null;
  }
}

export async function scrapeYoutube(
  youtubeUrl: string,
): Promise<YouTubeScraperResult> {
  if (!isYoutubeUrl(youtubeUrl)) {
    throw new Error("Invalid YouTube URL");
  }

  const cleanedUrl = cleanYoutubeUrl(youtubeUrl);

  const fromScrapeCreators = await fetchScrapeCreators(cleanedUrl);
  if (fromScrapeCreators && hasTranscript(fromScrapeCreators)) {
    return fromScrapeCreators;
  }

  const fromSupadata = await fetchSupadata(cleanedUrl);
  if (fromSupadata && hasTranscript(fromSupadata)) {
    return fromSupadata;
  }

  if (!getApiKey("SCRAPECREATORS_API_KEY") && !getApiKey("SUPADATA_API_KEY")) {
    throw new Error("No API keys found for Scrape Creators or Supadata");
  }

  throw new Error("Failed to fetch transcript from available providers");
}

export async function getTranscript(youtubeUrl: string): Promise<string> {
  const result = await scrapeYoutube(youtubeUrl);
  if (!hasTranscript(result)) {
    throw new Error("Video has no transcript");
  }

  const transcript = parsedTranscript(result);
  if (!transcript) {
    throw new Error("Transcript is empty");
  }

  return transcript;
}

export function formatYoutubeLoaderOutput(
  result: YouTubeScraperResult,
): string {
  const outputParts: string[] = [];

  if (result.title) {
    outputParts.push(`Title: ${result.title}`);
  }
  if (result.channel?.title) {
    outputParts.push(`Channel: ${result.channel.title}`);
  }
  if (result.durationFormatted) {
    outputParts.push(`Duration: ${result.durationFormatted}`);
  }
  if (result.publishDateText) {
    outputParts.push(`Published: ${result.publishDateText}`);
  }
  if (typeof result.viewCountInt === "number") {
    outputParts.push(`Views: ${result.viewCountInt.toLocaleString()}`);
  }
  if (typeof result.likeCountInt === "number") {
    outputParts.push(`Likes: ${result.likeCountInt.toLocaleString()}`);
  }

  outputParts.push("");

  if (result.description) {
    outputParts.push(`Description:\n${result.description}`);
    outputParts.push("");
  }

  const transcript = parsedTranscript(result);
  if (transcript) {
    outputParts.push(`Transcript:\n${transcript}`);
  } else {
    outputParts.push("Transcript: Not available for this video");
  }

  return outputParts.join("\n");
}
