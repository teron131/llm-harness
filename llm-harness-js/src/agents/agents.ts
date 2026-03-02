import type { ClientTool, ServerTool } from "@langchain/core/tools";
import Exa from "exa-js";
import { createAgent } from "langchain";
import type { ZodTypeAny, z } from "zod";

import { MediaMessage } from "../clients/multimodal.js";
import { ChatOpenRouter } from "../clients/openrouter.js";
import { webloaderTool } from "../tools/web/webloader.js";
import type { Summary } from "./youtube/schemas.js";
import { summarizeVideo as summarizeVideoLite } from "./youtube/summarizer.js";
import { summarizeVideo as summarizeVideoGemini } from "./youtube/summarizer-gemini.js";
import { summarizeVideo as summarizeVideoReact } from "./youtube/summarizer-react.js";

type GenericTool = ClientTool | ServerTool;
type AgentResponse<T extends ZodTypeAny | null> = {
  messages: unknown[];
  structuredResponse?: T extends ZodTypeAny ? z.output<T> : never;
};
type AgentOutput<T extends ZodTypeAny | null> = T extends ZodTypeAny
  ? z.output<T>
  : string;

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
      outputSchema: this.outputSchema,
    });

    return this.outputSchema.parse(result.answer);
  }
}

export class BaseHarnessAgent<T extends ZodTypeAny | null = null> {
  protected readonly agent: ReturnType<typeof createAgent>;
  protected readonly model: any;
  protected readonly responseFormat: T | undefined;

  constructor({
    model,
    temperature = 0,
    reasoningEffort = "medium",
    systemPrompt,
    responseFormat,
    tools = [],
    ...modelKwargs
  }: {
    model?: string;
    temperature?: number;
    reasoningEffort?: "minimal" | "low" | "medium" | "high";
    systemPrompt?: string;
    responseFormat?: T;
    tools?: readonly GenericTool[];
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

    const agentParams: {
      model: unknown;
      tools: GenericTool[];
      systemPrompt?: string;
      responseFormat?: T;
    } = {
      model: this.model,
      tools: [...tools],
    };
    if (systemPrompt !== undefined) {
      agentParams.systemPrompt = systemPrompt;
    }
    if (this.responseFormat) {
      agentParams.responseFormat = this.responseFormat;
    }

    this.agent = createAgent(agentParams as any);
  }

  protected processResponse(response: AgentResponse<T>): AgentOutput<T> {
    if (this.responseFormat) {
      if (response.structuredResponse === undefined) {
        throw new Error("Expected structuredResponse but none was returned.");
      }
      return response.structuredResponse as AgentOutput<T>;
    }

    const messageList = response.messages;
    const lastMessage =
      Array.isArray(messageList) && messageList.length > 0
        ? messageList.at(-1)
        : null;
    if (!lastMessage || typeof lastMessage !== "object") {
      return "" as AgentOutput<T>;
    }
    const content = (lastMessage as { content?: unknown }).content;
    return (content ?? "") as AgentOutput<T>;
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

  async invoke(userInput: string): Promise<AgentOutput<T>> {
    const response = (await this.agent.invoke({
      messages: [{ role: "user", content: userInput }],
    } as any)) as AgentResponse<T>;
    return this.processResponse(response);
  }
}

export class WebLoaderAgent<
  T extends ZodTypeAny | null = null,
> extends BaseHarnessAgent<T> {
  constructor(args: Record<string, unknown> = {}) {
    super({
      ...args,
      tools: [webloaderTool],
    });
  }

  async invoke(userInput: string): Promise<AgentOutput<T>> {
    const response = (await this.agent.invoke({
      messages: [{ role: "user", content: userInput }],
    } as any)) as AgentResponse<T>;
    return this.processResponse(response);
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
      tools: [webloaderTool],
      webSearch: true,
      webSearchEngine,
      webSearchMaxResults,
    });
  }

  async invoke(userInput: string): Promise<AgentOutput<T>> {
    const response = (await this.agent.invoke({
      messages: [{ role: "user", content: userInput }],
    } as any)) as AgentResponse<T>;
    return this.processResponse(response);
  }
}

export class ImageAnalysisAgent<
  T extends ZodTypeAny | null = null,
> extends BaseHarnessAgent<T> {
  async invoke(
    imagePaths: string | string[],
    description = "",
  ): Promise<AgentOutput<T>> {
    const mediaMessage = await MediaMessage.fromPathAsync({
      paths: imagePaths,
      description,
    });
    const response = (await this.agent.invoke({
      messages: [mediaMessage],
    } as any)) as AgentResponse<T>;
    return this.processResponse(response);
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
