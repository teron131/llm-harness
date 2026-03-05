import { fetchWithTimeout, nowEpochSeconds } from "../../utils.js";

const OPENROUTER_MODELS_URL = "https://openrouter.ai/api/frontend/models";
const OPENROUTER_THROUGHPUT_URL =
  "https://openrouter.ai/api/frontend/stats/throughput-comparison";
const OPENROUTER_LATENCY_URL =
  "https://openrouter.ai/api/frontend/stats/latency-comparison";
const OPENROUTER_E2E_LATENCY_URL =
  "https://openrouter.ai/api/frontend/stats/latency-e2e-comparison";
const OPENROUTER_EFFECTIVE_PRICING_URL =
  "https://openrouter.ai/api/frontend/stats/effective-pricing";

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_CONCURRENCY = 8;
const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_RETRY_BASE_DELAY_MS = 300;

type JsonObject = Record<string, unknown>;

type OpenRouterFrontendModel = {
  slug?: string | null;
  permaslug?: string | null;
};

type OpenRouterStatsPoint = {
  x?: string;
  y?: Record<string, number | null>;
};

type OpenRouterStatsResponse = {
  data?: OpenRouterStatsPoint[];
};

type OpenRouterModelStats = {
  throughput?: OpenRouterStatsResponse | null;
  latency?: OpenRouterStatsResponse | null;
  latency_e2e?: OpenRouterStatsResponse | null;
};

type OpenRouterEffectivePricingResponse = {
  data?: {
    weightedInputPrice?: number | null;
    weightedOutputPrice?: number | null;
  };
};

/**
 * Options for scraping OpenRouter performance stats for a selected model list.
 */
export type OpenRouterScraperOptions = {
  modelIds: string[];
  timeoutMs?: number;
  concurrency?: number;
  maxRetries?: number;
  retryBaseDelayMs?: number;
};

export type OpenRouterPerformanceSummary = {
  throughput_tokens_per_second_median: number | null;
  latency_seconds_median: number | null;
  e2e_latency_seconds_median: number | null;
};

export type OpenRouterPricingSummary = {
  weighted_input_price_per_1m: number | null;
  weighted_output_price_per_1m: number | null;
};

export type OpenRouterScrapedModel = {
  id: string;
  permaslug: string | null;
  performance: OpenRouterPerformanceSummary;
  pricing: OpenRouterPricingSummary;
};

export type OpenRouterScrapedPayload = {
  fetched_at_epoch_seconds: number;
  total_requested_models: number;
  total_resolved_models: number;
  models: OpenRouterScrapedModel[];
};

export type OpenRouterSingleModelOptions = Omit<
  OpenRouterScraperOptions,
  "modelIds"
>;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function sanitizeModelId(modelId: string): string {
  return (
    modelId
      .trim()
      .toLowerCase()
      // Normalize OpenRouter route suffixes (e.g. :free, :exacto) to base model id.
      .replace(/:[a-z0-9._-]+$/i, "")
  );
}

function asRecord(value: unknown): JsonObject {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
}

function asFiniteNumber(value: unknown): number | null {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : null;
}

function finiteNumbers(values: unknown[]): number[] {
  return values
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
}

function median(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[mid] ?? null;
  }
  const left = sorted[mid - 1];
  const right = sorted[mid];
  if (left == null || right == null) {
    return null;
  }
  return (left + right) / 2;
}

function average(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function toDailyAveragedValues(
  response: OpenRouterStatsResponse | null,
  scaleToSeconds: boolean,
): number[] {
  if (!response || !Array.isArray(response.data)) {
    return [];
  }
  return response.data
    .map((point) => {
      const y = asRecord(point.y);
      const values = finiteNumbers(Object.values(y));
      const dailyAverage = average(values);
      if (dailyAverage == null) {
        return null;
      }
      return scaleToSeconds ? dailyAverage / 1000 : dailyAverage;
    })
    .filter(
      (value): value is number => value != null && Number.isFinite(value),
    );
}

function summarizePerformance(
  stats: OpenRouterModelStats,
): OpenRouterPerformanceSummary {
  const throughputValues = toDailyAveragedValues(
    stats.throughput ?? null,
    false,
  );
  const latencyValues = toDailyAveragedValues(stats.latency ?? null, true);
  const e2eLatencyValues = toDailyAveragedValues(
    stats.latency_e2e ?? null,
    true,
  );

  return {
    throughput_tokens_per_second_median: median(throughputValues),
    latency_seconds_median: median(latencyValues),
    e2e_latency_seconds_median: median(e2eLatencyValues),
  };
}

function summarizePricing(
  response: OpenRouterEffectivePricingResponse | null,
): OpenRouterPricingSummary {
  const data = asRecord(response?.data);
  return {
    weighted_input_price_per_1m: asFiniteNumber(data.weightedInputPrice),
    weighted_output_price_per_1m: asFiniteNumber(data.weightedOutputPrice),
  };
}

function emptyScrapedModel(modelId: string): OpenRouterScrapedModel {
  return {
    id: modelId,
    permaslug: null,
    performance: summarizePerformance({}),
    pricing: summarizePricing(null),
  };
}

async function fetchJsonWithRetry<T>(
  url: string,
  timeoutMs: number,
  maxRetries: number,
  retryBaseDelayMs: number,
): Promise<T> {
  let lastError: unknown = null;

  for (let attempt = 0; attempt < maxRetries; attempt += 1) {
    try {
      const response = await fetchWithTimeout(url, {}, timeoutMs);
      if (!response.ok) {
        const status = response.status;
        if ((status === 429 || status >= 500) && attempt < maxRetries - 1) {
          const backoffMs =
            retryBaseDelayMs * 2 ** attempt + Math.floor(Math.random() * 100);
          await sleep(backoffMs);
          continue;
        }
        throw new Error(`OpenRouter request failed: ${status} (${url})`);
      }
      return (await response.json()) as T;
    } catch (error) {
      lastError = error;
      if (attempt < maxRetries - 1) {
        const backoffMs =
          retryBaseDelayMs * 2 ** attempt + Math.floor(Math.random() * 100);
        await sleep(backoffMs);
        continue;
      }
    }
  }

  throw lastError ?? new Error(`OpenRouter request failed: ${url}`);
}

async function mapWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  mapper: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const safeConcurrency = Math.max(1, Math.floor(concurrency));
  const results = new Array<R>(items.length);
  let cursor = 0;

  async function worker(): Promise<void> {
    for (;;) {
      const index = cursor;
      cursor += 1;
      if (index >= items.length) {
        return;
      }
      results[index] = await mapper(items[index] as T, index);
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(safeConcurrency, items.length) }, () =>
      worker(),
    ),
  );
  return results;
}

function buildPermaslugLookup(
  models: OpenRouterFrontendModel[],
): Map<string, string> {
  const permaslugBySlug = new Map<string, string>();
  for (const model of models) {
    if (typeof model.slug !== "string" || typeof model.permaslug !== "string") {
      continue;
    }
    const slug = sanitizeModelId(model.slug);
    const permaslug = model.permaslug.trim();
    if (!slug || !permaslug) {
      continue;
    }
    permaslugBySlug.set(slug, permaslug);
  }
  return permaslugBySlug;
}

function hasMeaningfulPerformance(
  performance: OpenRouterPerformanceSummary,
): boolean {
  return (
    performance.throughput_tokens_per_second_median != null ||
    performance.latency_seconds_median != null ||
    performance.e2e_latency_seconds_median != null
  );
}

function hasMeaningfulPricing(pricing: OpenRouterPricingSummary): boolean {
  const weightedInput = pricing.weighted_input_price_per_1m;
  const weightedOutput = pricing.weighted_output_price_per_1m;
  const hasInput = weightedInput != null && weightedInput > 0;
  const hasOutput = weightedOutput != null && weightedOutput > 0;
  return hasInput || hasOutput;
}

function splitSlugTokens(value: string): string[] {
  return value
    .toLowerCase()
    .split(/[-._/]+/)
    .filter(Boolean);
}

function tokenOverlapScore(
  targetTokens: string[],
  candidateTokens: string[],
): number {
  if (targetTokens.length === 0) {
    return 0;
  }
  const targetSet = new Set(targetTokens);
  const candidateSet = new Set(candidateTokens);
  const overlapCount = [...targetSet].filter((token) =>
    candidateSet.has(token),
  ).length;
  return overlapCount / targetSet.size;
}

function buildSlugFallbackCandidates(
  modelId: string,
  availableSlugs: string[],
): string[] {
  const normalized = sanitizeModelId(modelId);
  const [provider, modelName = ""] = normalized.split("/", 2);
  if (!provider || !modelName) {
    return [normalized];
  }

  const targetTokens = splitSlugTokens(modelName);
  const corePrefix = targetTokens.slice(0, 2).join("-");
  const scoredCandidates = availableSlugs
    .filter((slug) => slug.startsWith(`${provider}/`) && slug !== normalized)
    .map((slug) => {
      const candidateModel = slug.slice(provider.length + 1);
      const candidateTokens = splitSlugTokens(candidateModel);
      const overlapScore = tokenOverlapScore(targetTokens, candidateTokens);
      const prefixScore =
        corePrefix.length > 0 && candidateModel.startsWith(corePrefix)
          ? 0.2
          : 0;
      const score = overlapScore + prefixScore;
      return {
        slug,
        score,
        lengthDelta: Math.abs(candidateModel.length - modelName.length),
      };
    })
    .filter((candidate) => candidate.score >= 0.6)
    .sort((left, right) => {
      if (left.score !== right.score) {
        return right.score - left.score;
      }
      if (left.lengthDelta !== right.lengthDelta) {
        return left.lengthDelta - right.lengthDelta;
      }
      return left.slug.localeCompare(right.slug);
    })
    .slice(0, 8)
    .map((candidate) => candidate.slug);

  return [normalized, ...scoredCandidates];
}

async function fetchPerformanceForPermaslug(
  permaslug: string,
  timeoutMs: number,
  maxRetries: number,
  retryBaseDelayMs: number,
): Promise<{
  performance: OpenRouterModelStats;
  pricing: OpenRouterEffectivePricingResponse;
}> {
  const query = new URLSearchParams({ permaslug });
  const [throughput, latency, latencyE2e, effectivePricing] = await Promise.all(
    [
      fetchJsonWithRetry<OpenRouterStatsResponse>(
        `${OPENROUTER_THROUGHPUT_URL}?${query.toString()}`,
        timeoutMs,
        maxRetries,
        retryBaseDelayMs,
      ),
      fetchJsonWithRetry<OpenRouterStatsResponse>(
        `${OPENROUTER_LATENCY_URL}?${query.toString()}`,
        timeoutMs,
        maxRetries,
        retryBaseDelayMs,
      ),
      fetchJsonWithRetry<OpenRouterStatsResponse>(
        `${OPENROUTER_E2E_LATENCY_URL}?${query.toString()}`,
        timeoutMs,
        maxRetries,
        retryBaseDelayMs,
      ),
      fetchJsonWithRetry<OpenRouterEffectivePricingResponse>(
        `${OPENROUTER_EFFECTIVE_PRICING_URL}?${query.toString()}`,
        timeoutMs,
        maxRetries,
        retryBaseDelayMs,
      ),
    ],
  );

  return {
    performance: {
      throughput,
      latency,
      latency_e2e: latencyE2e,
    },
    pricing: effectivePricing,
  };
}

async function fetchBestAvailableModelStats(
  modelId: string,
  availableSlugs: string[],
  permaslugBySlug: Map<string, string>,
  timeoutMs: number,
  maxRetries: number,
  retryBaseDelayMs: number,
): Promise<OpenRouterScrapedModel> {
  const slugCandidates = buildSlugFallbackCandidates(modelId, availableSlugs);
  const permaslugCandidates = slugCandidates
    .map((slug) => permaslugBySlug.get(slug) ?? null)
    .filter(
      (slug): slug is string => typeof slug === "string" && slug.length > 0,
    );

  if (permaslugCandidates.length === 0) {
    return emptyScrapedModel(modelId);
  }

  let firstResolved: OpenRouterScrapedModel | null = null;
  for (const permaslug of permaslugCandidates) {
    try {
      const stats = await fetchPerformanceForPermaslug(
        permaslug,
        timeoutMs,
        maxRetries,
        retryBaseDelayMs,
      );
      const performance = summarizePerformance(stats.performance);
      const pricing = summarizePricing(stats.pricing);
      const resolvedModel: OpenRouterScrapedModel = {
        id: modelId,
        permaslug,
        performance,
        pricing,
      };
      firstResolved ??= resolvedModel;
      if (
        hasMeaningfulPerformance(performance) ||
        hasMeaningfulPricing(pricing)
      ) {
        return resolvedModel;
      }
    } catch {
      continue;
    }
  }

  return firstResolved ?? emptyScrapedModel(modelId);
}

/**
 * Scrape OpenRouter performance stats for a finalized set of model IDs.
 *
 * This intentionally avoids full-catalog scraping and only fetches stats for
 * `options.modelIds`.
 */
export async function getOpenRouterScrapedStats(
  options: OpenRouterScraperOptions,
): Promise<OpenRouterScrapedPayload> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const concurrency = options.concurrency ?? DEFAULT_CONCURRENCY;
  const maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
  const retryBaseDelayMs =
    options.retryBaseDelayMs ?? DEFAULT_RETRY_BASE_DELAY_MS;
  const uniqueModelIds = Array.from(
    new Set(options.modelIds.map((modelId) => modelId.trim()).filter(Boolean)),
  );

  const modelDirectory = await fetchJsonWithRetry<{
    data?: OpenRouterFrontendModel[];
  }>(OPENROUTER_MODELS_URL, timeoutMs, maxRetries, retryBaseDelayMs);
  const permaslugBySlug = buildPermaslugLookup(modelDirectory.data ?? []);
  const availableSlugs = [...permaslugBySlug.keys()];

  const models = await mapWithConcurrency(
    uniqueModelIds,
    concurrency,
    async (modelId) => {
      // Keep batch resilient and try close sibling slugs when a route is stale.
      return fetchBestAvailableModelStats(
        modelId,
        availableSlugs,
        permaslugBySlug,
        timeoutMs,
        maxRetries,
        retryBaseDelayMs,
      );
    },
  );

  return {
    fetched_at_epoch_seconds: nowEpochSeconds(),
    total_requested_models: uniqueModelIds.length,
    total_resolved_models: models.filter((model) => model.permaslug != null)
      .length,
    models,
  };
}

/**
 * Fetch OpenRouter performance stats for exactly one OpenRouter model ID.
 *
 * Example input:
 * - `openai/gpt-5.3-codex`
 * - `google/gemini-3.1-pro-preview`
 * - `meta-llama/llama-4-maverick:free` (free suffix is normalized)
 */
export async function getOpenRouterModelStats(
  modelId: string,
  options: OpenRouterSingleModelOptions = {},
): Promise<OpenRouterScrapedModel> {
  const scraperOptions: OpenRouterScraperOptions = {
    modelIds: [modelId],
    ...(options.timeoutMs != null ? { timeoutMs: options.timeoutMs } : {}),
    ...(options.concurrency != null
      ? { concurrency: options.concurrency }
      : {}),
    ...(options.maxRetries != null ? { maxRetries: options.maxRetries } : {}),
    ...(options.retryBaseDelayMs != null
      ? { retryBaseDelayMs: options.retryBaseDelayMs }
      : {}),
  };
  const payload = await getOpenRouterScrapedStats(scraperOptions);

  const firstModel = payload.models[0];
  if (!firstModel) {
    return emptyScrapedModel(modelId);
  }
  return firstModel;
}
