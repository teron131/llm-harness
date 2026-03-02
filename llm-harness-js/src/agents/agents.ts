import Exa from "exa-js";
import type { ZodTypeAny, z } from "zod";

import { MediaMessage } from "../clients/multimodal.js";
import { ChatOpenRouter } from "../clients/openrouter.js";
import { webloaderTool } from "../tools/web/webloader.js";
import type { Summary } from "./youtube/schemas.js";
import { summarizeVideo as summarizeVideoReact } from "./youtube/summarizer.js";
import { summarizeVideo as summarizeVideoGemini } from "./youtube/summarizer-gemini.js";
import { summarizeVideo as summarizeVideoLite } from "./youtube/summarizer-lite.js";

function extractUrls(text: string): string[] {
  const matches = text.match(/https?:\/\/[^\s)]+/g);
  return matches ?? [];
}

export class ExaAgent<T extends ZodTypeAny> {
  private readonly exa: any;
  private readonly systemPrompt: string;
  private readonly outputSchema: T;

  constructor(systemPrompt: string, outputSchema: T) {
    this.systemPrompt = systemPrompt;
    this.outputSchema = outputSchema;
    this.exa = new (Exa as any)(process.env.EXA_API_KEY);
  }

  async invoke(query: string): Promise<z.output<T>> {
    const result = await this.exa.answer(query, {
      systemPrompt: this.systemPrompt,
      text: true,
    });

    return this.outputSchema.parse(result.answer);
  }
}

export class BaseHarnessAgent<T extends ZodTypeAny | null = null> {
  protected readonly model: any;
  protected readonly responseFormat: T | undefined;

  constructor({
    model,
    temperature = 0,
    reasoningEffort = "medium",
    responseFormat,
    ...modelKwargs
  }: {
    model?: string;
    temperature?: number;
    reasoningEffort?: "minimal" | "low" | "medium" | "high";
    responseFormat?: T;
    [key: string]: unknown;
  }) {
    const modelName = model ?? process.env.FAST_LLM;
    if (!modelName) {
      throw new Error(
        "No model configured. Pass `model=...` or set `FAST_LLM`.",
      );
    }

    this.model = ChatOpenRouter({
      model: modelName,
      temperature,
      reasoningEffort,
      ...modelKwargs,
    });

    this.responseFormat = responseFormat;
  }

  protected async invokeModel(
    messages: Array<{ role: string; content: unknown }>,
  ): Promise<unknown> {
    if (this.responseFormat) {
      const structured = this.model.withStructuredOutput(this.responseFormat);
      return structured.invoke(messages);
    }

    const response = await this.model.invoke(messages);
    return response?.content ?? "";
  }
}

export class WebSearchAgent<
  T extends ZodTypeAny | null = null,
> extends BaseHarnessAgent<T> {
  constructor({
    webSearchEngine,
    webSearchMaxResults = 5,
    ...args
  }: {
    webSearchEngine?: "native" | "exa";
    webSearchMaxResults?: number;
  } & Record<string, unknown> = {}) {
    super({
      ...args,
      webSearch: true,
      webSearchEngine,
      webSearchMaxResults,
    });
  }

  invoke(userInput: string): Promise<unknown> {
    return this.invokeModel([{ role: "user", content: userInput }]);
  }
}

export class WebLoaderAgent<
  T extends ZodTypeAny | null = null,
> extends BaseHarnessAgent<T> {
  async invoke(userInput: string): Promise<unknown> {
    const urls = extractUrls(userInput);
    const loaded = urls.length ? await webloaderTool(urls) : [];
    const content = `${userInput}${loaded.length ? `\n\nLoaded URLs:\n${loaded.join("\n\n")}` : ""}`;
    return this.invokeModel([{ role: "user", content }]);
  }
}

export class WebSearchLoaderAgent<
  T extends ZodTypeAny | null = null,
> extends BaseHarnessAgent<T> {
  constructor({
    webSearchEngine,
    webSearchMaxResults = 5,
    ...args
  }: {
    webSearchEngine?: "native" | "exa";
    webSearchMaxResults?: number;
  } & Record<string, unknown> = {}) {
    super({
      ...args,
      webSearch: true,
      webSearchEngine,
      webSearchMaxResults,
    });
  }

  async invoke(userInput: string): Promise<unknown> {
    const urls = extractUrls(userInput);
    const loaded = urls.length ? await webloaderTool(urls) : [];
    const content = `${userInput}${loaded.length ? `\n\nLoaded URLs:\n${loaded.join("\n\n")}` : ""}`;
    return this.invokeModel([{ role: "user", content }]);
  }
}

export class ImageAnalysisAgent<
  T extends ZodTypeAny | null = null,
> extends BaseHarnessAgent<T> {
  async invoke(
    imagePaths: string | string[],
    description = "",
  ): Promise<unknown> {
    const mediaMessage = await MediaMessage.fromPathAsync({
      paths: imagePaths,
      description,
    });
    return this.invokeModel([mediaMessage]);
  }
}

export class YouTubeSummarizerReAct {
  private readonly targetLanguage: string | null | undefined;

  constructor(targetLanguage?: string | null) {
    this.targetLanguage = targetLanguage;
  }

  invoke(transcriptOrUrl: string): Promise<Summary> {
    return summarizeVideoReact({
      transcriptOrUrl,
      targetLanguage: this.targetLanguage ?? null,
    });
  }
}

export class YouTubeSummarizer {
  private readonly targetLanguage: string | null | undefined;

  constructor(targetLanguage?: string | null) {
    this.targetLanguage = targetLanguage;
  }

  invoke(transcriptOrUrl: string): Promise<Summary> {
    return summarizeVideoLite({
      transcriptOrUrl,
      targetLanguage: this.targetLanguage ?? null,
    });
  }
}

export class YouTubeSummarizerGemini {
  private readonly options: {
    model?: string;
    thinkingLevel?: "minimal" | "low" | "medium" | "high";
    targetLanguage?: string;
    apiKey?: string;
  };

  constructor(
    options: {
      model?: string;
      thinkingLevel?: "minimal" | "low" | "medium" | "high";
      targetLanguage?: string;
      apiKey?: string;
    } = {},
  ) {
    this.options = options;
  }

  invoke(videoUrl: string): Promise<Summary | null> {
    const payload: {
      videoUrl: string;
      model?: string;
      thinkingLevel?: "minimal" | "low" | "medium" | "high";
      targetLanguage?: string;
      apiKey?: string;
    } = { videoUrl };

    if (this.options.model) {
      payload.model = this.options.model;
    }
    if (this.options.thinkingLevel) {
      payload.thinkingLevel = this.options.thinkingLevel;
    }
    if (this.options.targetLanguage) {
      payload.targetLanguage = this.options.targetLanguage;
    }
    if (this.options.apiKey) {
      payload.apiKey = this.options.apiKey;
    }

    return summarizeVideoGemini(payload);
  }
}

export const YouTubeSummarizerReActAgent = YouTubeSummarizerReAct;
export const YouTubeSummarizerLiteAgent = YouTubeSummarizer;
export const YouTubeSummarizerGeminiAgent = YouTubeSummarizerGemini;
