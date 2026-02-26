import { GoogleGenAI, Type } from "@google/genai";

import { trackUsage } from "../../clients/usage.js";
import { getGeminiSummaryPrompt } from "./prompts.js";
import { Summary, SummarySchema } from "./schemas.js";

const DEFAULT_MODEL = "gemini-3-flash-preview";

const USD_PER_M_TOKENS_BY_MODEL: Record<
  string,
  { input: number; output: number }
> = {
  "gemini-3-flash-preview": { input: 0.5, output: 3 },
  "gemini-3-pro-preview": { input: 2, output: 12 },
};

function calculateCost(
  model: string,
  promptTokens: number,
  totalTokens: number,
): number {
  const pricing = USD_PER_M_TOKENS_BY_MODEL[model];
  if (!pricing) {
    return 0;
  }
  const outputTokens = Math.max(0, totalTokens - promptTokens);
  return (
    (promptTokens / 1_000_000) * pricing.input +
    (outputTokens / 1_000_000) * pricing.output
  );
}

export async function analyzeVideoUrl({
  videoUrl,
  model = DEFAULT_MODEL,
  thinkingLevel = "medium",
  targetLanguage = "auto",
  apiKey,
  timeout = 600,
}: {
  videoUrl: string;
  model?: string;
  thinkingLevel?: "minimal" | "low" | "medium" | "high";
  targetLanguage?: string;
  apiKey?: string;
  timeout?: number;
}): Promise<Summary | null> {
  const resolvedApiKey =
    apiKey ?? process.env.GOOGLE_API_KEY ?? process.env.GEMINI_API_KEY;
  if (!resolvedApiKey) {
    throw new Error("API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY");
  }

  const client = new GoogleGenAI({ apiKey: resolvedApiKey });

  try {
    const response = await client.models.generateContent({
      model,
      contents: [
        {
          fileData: {
            fileUri: videoUrl,
          },
        },
        {
          text: getGeminiSummaryPrompt(targetLanguage),
        },
      ],
      config: {
        httpOptions: { timeout: timeout * 1000 },
        thinkingConfig: { thinkingBudget: thinkingLevel === "high" ? -1 : 0 },
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            overview: { type: Type.STRING },
            chapters: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  title: { type: Type.STRING },
                  description: { type: Type.STRING },
                  start_time: { type: Type.STRING, nullable: true },
                  end_time: { type: Type.STRING, nullable: true },
                },
                required: ["title", "description"],
              },
            },
          },
          required: ["overview", "chapters"],
        },
      },
    });

    const text = response.text;
    if (!text) {
      return null;
    }

    const analysis = SummarySchema.parse(JSON.parse(text));

    const usage = response.usageMetadata;
    if (
      usage?.promptTokenCount !== undefined &&
      usage?.totalTokenCount !== undefined
    ) {
      const cost = calculateCost(
        model,
        usage.promptTokenCount,
        usage.totalTokenCount,
      );
      trackUsage(
        usage.promptTokenCount,
        usage.totalTokenCount - usage.promptTokenCount,
        cost,
      );
    }

    return analysis;
  } catch {
    return null;
  }
}

export async function summarizeVideo({
  videoUrl,
  model = DEFAULT_MODEL,
  thinkingLevel = "medium",
  targetLanguage = "auto",
  apiKey,
}: {
  videoUrl: string;
  model?: string;
  thinkingLevel?: "minimal" | "low" | "medium" | "high";
  targetLanguage?: string;
  apiKey?: string;
}): Promise<Summary | null> {
  const payload: {
    videoUrl: string;
    model?: string;
    thinkingLevel?: "minimal" | "low" | "medium" | "high";
    targetLanguage?: string;
    apiKey?: string;
  } = {
    videoUrl,
    model,
    thinkingLevel,
    targetLanguage,
  };

  if (apiKey) {
    payload.apiKey = apiKey;
  }

  return analyzeVideoUrl(payload);
}
