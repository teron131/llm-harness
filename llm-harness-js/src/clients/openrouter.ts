import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenRouter as NativeChatOpenRouter } from "@langchain/openrouter";

const INVALID_MODEL_FORMAT_MESSAGE =
  "Invalid OpenRouter model format: {model}. Expected PROVIDER/MODEL";

type PluginConfig = Record<string, unknown>;

function isOpenRouter(model: string): boolean {
  return model.includes("/") && model.split("/").length === 2;
}

function validateOpenRouterModel(model: string): void {
  if (!isOpenRouter(model)) {
    throw new Error(INVALID_MODEL_FORMAT_MESSAGE.replace("{model}", model));
  }
}

function buildDerivedPlugins({
  webSearch,
  webSearchEngine,
  webSearchMaxResults,
  pdfEngine,
}: {
  webSearch: boolean;
  webSearchEngine?: "native" | "exa" | undefined;
  webSearchMaxResults: number;
  pdfEngine?: "mistral-ocr" | "pdf-text" | "native" | undefined;
}): PluginConfig[] {
  const plugins: PluginConfig[] = [];

  if (pdfEngine) {
    plugins.push({ id: "file-parser", pdf: { engine: pdfEngine } });
  }

  if (webSearch) {
    const webPlugin: PluginConfig = { id: "web" };
    if (webSearchEngine) {
      webPlugin.engine = webSearchEngine;
    }
    if (webSearchMaxResults !== 5) {
      webPlugin.max_results = webSearchMaxResults;
    }
    plugins.push(webPlugin);
  }

  return plugins;
}

function mergePlugins(
  derivedPlugins: PluginConfig[],
  explicitPlugins: unknown,
): PluginConfig[] {
  const byId = new Map<string, PluginConfig>();

  derivedPlugins.forEach((plugin) => {
    const id = typeof plugin.id === "string" ? plugin.id : null;
    if (id) {
      byId.set(id, plugin);
    }
  });

  if (Array.isArray(explicitPlugins)) {
    explicitPlugins.forEach((plugin, idx) => {
      if (!plugin || typeof plugin !== "object") {
        return;
      }
      const typed = plugin as PluginConfig;
      const id = typeof typed.id === "string" ? typed.id : `__anon_${idx}`;
      byId.set(id, typed);
    });
  }

  return [...byId.values()];
}

function buildProviderPreferences({
  providerSort,
  openrouterProvider,
}: {
  providerSort: "throughput" | "price" | "latency";
  openrouterProvider?: unknown;
}): PluginConfig {
  const provider =
    openrouterProvider && typeof openrouterProvider === "object"
      ? { ...(openrouterProvider as PluginConfig) }
      : {};
  if (!provider.sort) {
    provider.sort = providerSort;
  }
  return provider;
}

export function ChatOpenRouter({
  model,
  temperature = 0.7,
  reasoningEffort,
  providerSort = "throughput",
  webSearch = false,
  webSearchEngine,
  webSearchMaxResults = 5,
  pdfEngine,
  ...kwargs
}: {
  model: string;
  temperature?: number;
  reasoningEffort?: "minimal" | "low" | "medium" | "high";
  providerSort?: "throughput" | "price" | "latency";
  webSearch?: boolean;
  webSearchEngine?: "native" | "exa";
  webSearchMaxResults?: number;
  pdfEngine?: "mistral-ocr" | "pdf-text" | "native";
  [key: string]: unknown;
}): NativeChatOpenRouter {
  validateOpenRouterModel(model);

  const explicitPlugins = kwargs.plugins;
  delete kwargs.plugins;

  const openrouterProvider = kwargs.openrouter_provider;
  delete kwargs.openrouter_provider;

  delete kwargs.model;
  delete kwargs.temperature;

  const mergedPlugins = mergePlugins(
    buildDerivedPlugins({
      webSearch,
      webSearchEngine,
      webSearchMaxResults,
      pdfEngine,
    }),
    explicitPlugins,
  );

  const provider = buildProviderPreferences({
    providerSort,
    openrouterProvider,
  });

  const modelKwargs: Record<string, unknown> = { ...kwargs };
  if (reasoningEffort) {
    modelKwargs.reasoning = { effort: reasoningEffort };
  }

  const options: any = {
    model,
    temperature,
    modelKwargs,
  };
  if (Object.keys(provider).length > 0) {
    options.provider = provider;
  }
  if (mergedPlugins.length > 0) {
    options.plugins = mergedPlugins;
  }

  if (process.env.OPENROUTER_API_KEY) {
    options.apiKey = process.env.OPENROUTER_API_KEY;
  }

  return new NativeChatOpenRouter(options);
}

export function OpenRouterEmbeddings({
  model = "openai/text-embedding-3-small",
  ...kwargs
}: {
  model?: string;
  [key: string]: unknown;
} = {}): OpenAIEmbeddings {
  validateOpenRouterModel(model);

  const options: any = {
    model,
    configuration: {
      baseURL: "https://openrouter.ai/api/v1",
    },
    ...(kwargs as Record<string, unknown>),
  };

  if (process.env.OPENROUTER_API_KEY) {
    options.apiKey = process.env.OPENROUTER_API_KEY;
  }

  return new OpenAIEmbeddings(options);
}
