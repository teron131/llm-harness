import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { getArtificialAnalysisStats } from "./data-sources/llm/artificial-analysis-api";
import { getArtificialAnalysisScrapedEvalsOnlyStats } from "./data-sources/llm/artificial-analysis-scraper";
import {
  getMatchModelMapping,
  getScraperFallbackMatchDiagnostics,
} from "./data-sources/llm/matcher";
import { getModelsDevStats } from "./data-sources/llm/models-dev";
import { getOpenRouterScrapedStats } from "./data-sources/llm/openrouter-scraper";
import { percentileRank } from "./data-sources/utils";

const DEFAULT_OUTPUT_PATH = resolve(".cache/model_stats.json");
const CACHE_DIR = resolve(".cache");
const CACHE_TTL_SECONDS = 60 * 60 * 24;
const OPENROUTER_SPEED_CONCURRENCY = 8;
const NULL_FIELD_PRUNE_THRESHOLD = 0.5;
const NULL_FIELD_PRUNE_RECENT_LOOKBACK_DAYS = 90;
const MIN_INTELLIGENCE_COST_TOKEN_THRESHOLD = 1_000_000;
const INTELLIGENCE_COST_TOTAL_COST_KEY = "intelligence_index_cost_total_cost";
const INTELLIGENCE_COST_TOTAL_TOKENS_KEY =
  "intelligence_index_cost_total_tokens";
const INTELLIGENCE_BENCHMARK_KEYS = [
  "omniscience_accuracy",
  "hle",
  "lcr",
  "scicode",
] as const;
const AGENTIC_BENCHMARK_KEYS = [
  "omniscience_nonhallucination_rate",
  "gdpval_normalized",
  "ifbench",
  "terminalbench_hard",
] as const;
const DEFAULT_SPEED_OUTPUT_TOKEN_ANCHORS = [500, 2_000, 5_000, 10_000] as const;
const SPEED_OUTPUT_TOKEN_RANGE_MIN = 200;
const SPEED_OUTPUT_TOKEN_RANGE_MAX = 8_000;
const STABLE_TOP_LEVEL_KEYS = new Set<string>([
  "id",
  "name",
  "provider",
  "logo",
  "attachment",
  "reasoning",
  "release_date",
  "modalities",
  "open_weights",
  "cost",
  "context_window",
  "speed",
  "intelligence",
  "intelligence_index_cost",
  "evaluations",
  "scores",
  "percentiles",
]);
const EMPTY_OPENROUTER_SPEED = {
  throughput_tokens_per_second_median: null,
  latency_seconds_median: null,
  e2e_latency_seconds_median: null,
} as const;
const EMPTY_OPENROUTER_PRICING = {
  weighted_input: null,
  weighted_output: null,
} as const;
const SPEED_ANCHOR_QUANTILES = [0.25, 0.5, 0.75] as const;
const WEIGHTED_PRICE_INPUT_RATIO = 0.75;
const WEIGHTED_PRICE_OUTPUT_RATIO = 0.25;

type JsonObject = Record<string, unknown>;
type ModelsDevModel = Awaited<
  ReturnType<typeof getModelsDevStats>
>["models"][number];
type ArtificialAnalysisModel = Awaited<
  ReturnType<typeof getArtificialAnalysisStats>
>["models"][number];
type ScrapedEvalModel = {
  model_id?: unknown;
  logo?: unknown;
  evaluations?: unknown;
  intelligence?: unknown;
  intelligence_index_cost?: unknown;
};

const PRIMARY_PROVIDER_FILTER = "openrouter" as const;
const FALLBACK_PROVIDER_FILTERS = new Set(["openai", "google", "anthropic"]);

/**
 * Final selected model row exposed by the stats API.
 */
export type ModelStatsSelectedModel = {
  id: string | null;
  name: string | null;
  provider: string | null;
  logo: string;
  attachment: boolean | null;
  reasoning: boolean | null;
  release_date: string | null;
  modalities: unknown;
  open_weights: boolean | null;
  cost: unknown;
  context_window: unknown;
  speed: JsonObject;
  intelligence: unknown;
  intelligence_index_cost: unknown;
  evaluations: unknown;
  scores: unknown;
  percentiles: unknown;
};

/**
 * Final model stats payload returned by the public stats API.
 *
 * `fetched_at_epoch_seconds` is `null` when the upstream pipeline fails and
 * the function returns an empty-safe payload.
 */
export type ModelStatsSelectedPayload = {
  fetched_at_epoch_seconds: number | null;
  models: ModelStatsSelectedModel[];
};

/**
 * Options for model stats lookup.
 *
 * - omit `id` (or set `null`) to return the full list
 * - set `id` to return exact-id matches only
 */
export type ModelStatsSelectedOptions = {
  id?: string | null;
  apiKey?: string;
};

function providerFromId(modelId: unknown): string | null {
  if (typeof modelId !== "string") {
    return null;
  }
  const slashIndex = modelId.indexOf("/");
  if (slashIndex <= 0) {
    return null;
  }
  return modelId.slice(0, slashIndex);
}

function canonicalModelId(
  modelId: unknown,
  providerId: unknown,
  fallbackModelId: unknown,
): string | null {
  if (typeof modelId === "string" && modelId.includes("/")) {
    return modelId;
  }
  if (typeof providerId === "string" && typeof modelId === "string") {
    return `${providerId}/${modelId}`;
  }
  if (typeof providerId === "string" && typeof fallbackModelId === "string") {
    return `${providerId}/${fallbackModelId}`;
  }
  return typeof modelId === "string" ? modelId : null;
}

function providerFromModel(model: JsonObject): string | null {
  const fromId = providerFromId(model.id);
  if (fromId) {
    return fromId;
  }
  return typeof model.provider_id === "string" ? model.provider_id : null;
}

function asRecord(value: unknown): JsonObject {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
}

function asFiniteNumber(value: unknown): number | null {
  if (value == null) {
    return null;
  }
  if (typeof value === "string" && value.trim().length === 0) {
    return null;
  }
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : null;
}

function normalizeProviderModelId(id: string): string {
  const slashIndex = id.indexOf("/");
  if (slashIndex <= 0) {
    return id.toLowerCase().replace(/\./g, "-").replace(/-+/g, "-");
  }
  const provider = id.slice(0, slashIndex).toLowerCase();
  const model = id
    .slice(slashIndex + 1)
    .toLowerCase()
    .replace(/\./g, "-")
    .replace(/-+/g, "-");
  return `${provider}/${model}`;
}

function meanOfFinite(values: Array<number | null>): number | null {
  const finiteValues = values.filter(
    (value): value is number => value != null && Number.isFinite(value),
  );
  if (finiteValues.length === 0) {
    return null;
  }
  const total = finiteValues.reduce((sum, value) => sum + value, 0);
  return total / finiteValues.length;
}

function buildEvaluations(model: JsonObject): unknown {
  const evaluations = asRecord(model.evaluations);
  return Object.keys(evaluations).length > 0 ? evaluations : null;
}

function buildIntelligence(model: JsonObject): unknown {
  const intelligence = asRecord(model.intelligence);
  const nonhallucinationRate = asFiniteNumber(
    intelligence.omniscience_hallucination_rate,
  );
  if (nonhallucinationRate != null) {
    intelligence.omniscience_nonhallucination_rate = nonhallucinationRate;
    delete intelligence.omniscience_hallucination_rate;
  }
  delete intelligence[INTELLIGENCE_COST_TOTAL_COST_KEY];
  delete intelligence[INTELLIGENCE_COST_TOTAL_TOKENS_KEY];
  return Object.keys(intelligence).length > 0 ? intelligence : null;
}

function buildIntelligenceIndexCost(model: JsonObject): unknown {
  const fromRow = asRecord(model.intelligence_index_cost);
  const fromIntelligence = asRecord(model.intelligence);
  const totalCost =
    asFiniteNumber(fromRow.total_cost) ??
    asFiniteNumber(fromIntelligence[INTELLIGENCE_COST_TOTAL_COST_KEY]);
  const totalTokens =
    asFiniteNumber(fromRow.total_tokens) ??
    asFiniteNumber(fromIntelligence[INTELLIGENCE_COST_TOTAL_TOKENS_KEY]);
  const normalized = {
    ...fromRow,
    total_cost: totalCost,
    total_tokens:
      totalTokens != null &&
      totalTokens >= MIN_INTELLIGENCE_COST_TOKEN_THRESHOLD
        ? totalTokens
        : null,
  } as JsonObject;
  const cleaned = Object.fromEntries(
    Object.entries(normalized).filter(([, value]) => value != null),
  );
  return Object.keys(cleaned).length > 0 ? cleaned : null;
}

function metricValue(model: JsonObject, key: string): number | null {
  const intelligence = asRecord(model.intelligence);
  const evaluations = asRecord(model.evaluations);
  if (key === "omniscience_nonhallucination_rate") {
    const nonhallucinationRate = asFiniteNumber(
      intelligence.omniscience_nonhallucination_rate,
    );
    if (nonhallucinationRate != null) {
      return nonhallucinationRate;
    }
    const nonhallucinationRateFromLegacyKey = asFiniteNumber(
      intelligence.omniscience_hallucination_rate,
    );
    return nonhallucinationRateFromLegacyKey;
  }
  return (
    asFiniteNumber(intelligence[key]) ??
    asFiniteNumber(evaluations[key]) ??
    null
  );
}

function blendedPriceValue(model: JsonObject): number | null {
  const cost = asRecord(model.cost);
  const inputCost = asFiniteNumber(cost.input);
  const outputCost = asFiniteNumber(cost.output);
  const weightedInputCost = asFiniteNumber(cost.weighted_input);
  const weightedOutputCost = asFiniteNumber(cost.weighted_output);
  const cacheReadCost = asFiniteNumber(cost.cache_read);
  const cacheWriteCost = asFiniteNumber(cost.cache_write);
  if (
    inputCost == null ||
    outputCost == null ||
    inputCost <= 0 ||
    outputCost <= 0
  ) {
    return null;
  }
  if (weightedInputCost != null || weightedOutputCost != null) {
    const effectiveInputCost =
      weightedInputCost != null ? weightedInputCost : inputCost;
    const effectiveOutputCost =
      weightedOutputCost != null ? weightedOutputCost : outputCost;
    return (
      WEIGHTED_PRICE_INPUT_RATIO * effectiveInputCost +
      WEIGHTED_PRICE_OUTPUT_RATIO * effectiveOutputCost
    );
  }

  const cacheWeightedInput = cacheReadCost != null ? cacheReadCost : inputCost;
  const cacheWeightedOutput =
    cacheWriteCost != null
      ? 0.1 * cacheWriteCost + 0.9 * outputCost
      : outputCost;
  const baseProxy =
    0.9 * (0.75 * cacheWeightedInput + 0.25 * inputCost) +
    0.1 * cacheWeightedOutput;

  const over200kCost = asRecord(cost.context_over_200k);
  const over200kInput = asFiniteNumber(over200kCost.input);
  const over200kOutput = asFiniteNumber(over200kCost.output);
  const over200kCacheRead = asFiniteNumber(over200kCost.cache_read);
  const over200kCacheWrite = asFiniteNumber(over200kCost.cache_write);
  if (
    over200kInput == null ||
    over200kOutput == null ||
    over200kInput <= 0 ||
    over200kOutput <= 0
  ) {
    return baseProxy;
  }

  const over200kInputWeighted =
    over200kCacheRead != null ? over200kCacheRead : over200kInput;
  const over200kOutputWeighted =
    over200kCacheWrite != null
      ? 0.1 * over200kCacheWrite + 0.9 * over200kOutput
      : over200kOutput;
  const over200kProxy =
    0.9 * (0.75 * over200kInputWeighted + 0.25 * over200kInput) +
    0.1 * over200kOutputWeighted;

  return 0.95 * baseProxy + 0.05 * over200kProxy;
}

function quantileFromSorted(values: number[], quantile: number): number | null {
  if (values.length === 0) {
    return null;
  }
  if (values.length === 1) {
    return values[0] ?? null;
  }
  const clampedQuantile = Math.min(1, Math.max(0, quantile));
  const index = (values.length - 1) * clampedQuantile;
  const lowerIndex = Math.floor(index);
  const upperIndex = Math.ceil(index);
  if (lowerIndex === upperIndex) {
    return values[lowerIndex] ?? null;
  }
  const lowerValue = values[lowerIndex];
  const upperValue = values[upperIndex];
  if (lowerValue == null || upperValue == null) {
    return null;
  }
  const ratio = index - lowerIndex;
  return lowerValue + (upperValue - lowerValue) * ratio;
}

function deriveSpeedOutputTokenAnchors(
  openRouterSpeedById: Map<string, JsonObject>,
): number[] {
  const impliedTokenUsages = Array.from(openRouterSpeedById.values())
    .map((speed) => {
      const throughputTokensPerSecond = asFiniteNumber(
        speed.throughput_tokens_per_second_median,
      );
      const latencySeconds = asFiniteNumber(speed.latency_seconds_median);
      const e2eLatencySeconds = asFiniteNumber(
        speed.e2e_latency_seconds_median,
      );
      if (
        throughputTokensPerSecond == null ||
        throughputTokensPerSecond <= 0 ||
        latencySeconds == null ||
        e2eLatencySeconds == null
      ) {
        return null;
      }
      const generationSeconds = e2eLatencySeconds - latencySeconds;
      if (generationSeconds <= 0) {
        return null;
      }
      return generationSeconds * throughputTokensPerSecond;
    })
    .filter((value): value is number => value != null && Number.isFinite(value))
    .sort((left, right) => left - right);

  if (impliedTokenUsages.length === 0) {
    return [...DEFAULT_SPEED_OUTPUT_TOKEN_ANCHORS];
  }

  const q0 = impliedTokenUsages[0] ?? null;
  const [q1, q2, q3] = SPEED_ANCHOR_QUANTILES.map((quantile) =>
    quantileFromSorted(impliedTokenUsages, quantile),
  );
  const q4 = impliedTokenUsages[impliedTokenUsages.length - 1] ?? null;
  const numericQuantileAnchors = [q0, q1, q2, q3, q4].filter(
    (value): value is number => value != null && Number.isFinite(value),
  );
  if (numericQuantileAnchors.length !== 5) {
    return [...DEFAULT_SPEED_OUTPUT_TOKEN_ANCHORS];
  }

  const sourceMin = numericQuantileAnchors[0] as number;
  const sourceMax = numericQuantileAnchors[
    numericQuantileAnchors.length - 1
  ] as number;
  if (!(sourceMax > sourceMin)) {
    return [...DEFAULT_SPEED_OUTPUT_TOKEN_ANCHORS];
  }

  return numericQuantileAnchors.map((anchor) => {
    const normalized = (anchor - sourceMin) / (sourceMax - sourceMin);
    const mapped =
      SPEED_OUTPUT_TOKEN_RANGE_MIN +
      normalized *
        (SPEED_OUTPUT_TOKEN_RANGE_MAX - SPEED_OUTPUT_TOKEN_RANGE_MIN);
    return Math.round(mapped);
  });
}

function buildScores(
  model: JsonObject,
  speed: JsonObject,
  speedOutputTokenAnchors: number[],
): unknown {
  const intelligenceIndex =
    metricValue(model, "intelligence_index") ??
    metricValue(model, "artificial_analysis_intelligence_index");
  const agenticIndex =
    metricValue(model, "agentic_index") ??
    metricValue(model, "artificial_analysis_agentic_index");
  const intelligenceBenchmarkMean = meanOfFinite(
    INTELLIGENCE_BENCHMARK_KEYS.map((key) => metricValue(model, key)),
  );
  const agenticBenchmarkMean = meanOfFinite(
    AGENTIC_BENCHMARK_KEYS.map((key) => metricValue(model, key)),
  );
  const intelligenceScore =
    intelligenceIndex != null && intelligenceBenchmarkMean != null
      ? (intelligenceIndex + intelligenceBenchmarkMean) / 2
      : null;
  const agenticScore =
    agenticIndex != null && agenticBenchmarkMean != null
      ? (agenticIndex + agenticBenchmarkMean) / 2
      : null;
  const blendedPrice = blendedPriceValue(model);
  const latencySeconds = asFiniteNumber(speed.latency_seconds_median);
  const throughputTokensPerSecond = asFiniteNumber(
    speed.throughput_tokens_per_second_median,
  );
  const e2eLatencySeconds = asFiniteNumber(speed.e2e_latency_seconds_median);
  const priceScore =
    blendedPrice != null && blendedPrice > 0 ? 1 / blendedPrice : null;
  const imaginedSpeedScore = meanOfFinite(
    speedOutputTokenAnchors.map((targetTokens) =>
      latencySeconds != null &&
      throughputTokensPerSecond != null &&
      throughputTokensPerSecond > 0
        ? targetTokens /
          (latencySeconds + targetTokens / throughputTokensPerSecond)
        : null,
    ),
  );
  const sortedAnchors = [...speedOutputTokenAnchors].sort(
    (left, right) => left - right,
  );
  const representativeTargetTokens = quantileFromSorted(sortedAnchors, 0.5);
  const observedE2eSpeedScore =
    representativeTargetTokens != null &&
    e2eLatencySeconds != null &&
    e2eLatencySeconds > 0
      ? representativeTargetTokens / e2eLatencySeconds
      : null;
  const speedScore = meanOfFinite([imaginedSpeedScore, observedE2eSpeedScore]);
  if (
    intelligenceScore == null &&
    agenticScore == null &&
    priceScore == null &&
    speedScore == null
  ) {
    return null;
  }
  return {
    intelligence_score: intelligenceScore,
    agentic_score: agenticScore,
    speed_score: speedScore,
    price_score: priceScore,
  };
}

function withComputedPercentiles(
  models: ModelStatsSelectedModel[],
): ModelStatsSelectedModel[] {
  const intelligenceScores = models.map((model) =>
    asFiniteNumber(asRecord(model.scores).intelligence_score),
  );
  const agenticScores = models.map((model) =>
    asFiniteNumber(asRecord(model.scores).agentic_score),
  );
  const speedScores = models.map((model) =>
    asFiniteNumber(asRecord(model.scores).speed_score),
  );
  const priceScores = models.map((model) =>
    asFiniteNumber(asRecord(model.scores).price_score),
  );

  return models.map((model) => {
    const scores = asRecord(model.scores);
    const intelligenceScore = asFiniteNumber(scores.intelligence_score);
    const agenticScore = asFiniteNumber(scores.agentic_score);
    const speedScore = asFiniteNumber(scores.speed_score);
    const priceScore = asFiniteNumber(scores.price_score);
    const intelligencePercentile =
      intelligenceScore == null
        ? null
        : percentileRank(intelligenceScores, intelligenceScore);
    const agenticPercentile =
      agenticScore == null ? null : percentileRank(agenticScores, agenticScore);
    const speedPercentile =
      speedScore == null ? null : percentileRank(speedScores, speedScore);
    const pricePercentile =
      priceScore == null ? null : percentileRank(priceScores, priceScore);
    return {
      ...model,
      percentiles: {
        intelligence_percentile: intelligencePercentile,
        agentic_percentile: agenticPercentile,
        speed_percentile: speedPercentile,
        price_percentile: pricePercentile,
      },
    };
  });
}

function normalize(value: string): string {
  return value
    .toLowerCase()
    .replace(/[._:\s]+/g, "-")
    .replace(/[^a-z0-9/-]+/g, "")
    .replace(/-+/g, "-")
    .replace(/^[-/]+|[-/]+$/g, "");
}

function creatorSlugFromApiModel(
  model: ArtificialAnalysisModel,
): string | null {
  const creator = asRecord(model.model_creator);
  if (typeof creator.slug === "string" && creator.slug.length > 0) {
    return creator.slug;
  }
  if (typeof creator.name === "string" && creator.name.length > 0) {
    return normalize(creator.name);
  }
  return null;
}

function artificialAnalysisModelId(
  model: ArtificialAnalysisModel,
): string | null {
  const creatorSlug = creatorSlugFromApiModel(model);
  const modelSlug = typeof model.slug === "string" ? model.slug : null;
  if (!creatorSlug || !modelSlug) {
    return null;
  }
  return `${creatorSlug}/${modelSlug}`;
}

function modelSlugFromModelId(modelId: unknown): string | null {
  if (typeof modelId !== "string" || modelId.length === 0) {
    return null;
  }
  const slug = modelId.split("/").at(-1);
  return slug && slug.length > 0 ? slug : null;
}

const MODEL_VARIANT_TOKENS = [
  "flash-lite",
  "flash",
  "pro",
  "nano",
  "mini",
  "lite",
] as const;

function hasToken(id: string, token: (typeof MODEL_VARIANT_TOKENS)[number]) {
  return id.includes(token);
}

function hasVariantConflict(
  artificialAnalysisSlug: string,
  matchedModelId: string,
): boolean {
  const aa = normalizeProviderModelId(artificialAnalysisSlug);
  const matched = normalizeProviderModelId(matchedModelId);
  return MODEL_VARIANT_TOKENS.some(
    (token) => hasToken(aa, token) !== hasToken(matched, token),
  );
}

function scopeToPreferredProviderModels(
  modelsDevModels: ModelsDevModel[],
): ModelsDevModel[] {
  const preferredModels = modelsDevModels.filter(
    (modelStatsModel) =>
      modelStatsModel.provider_id === PRIMARY_PROVIDER_FILTER ||
      FALLBACK_PROVIDER_FILTERS.has(modelStatsModel.provider_id),
  );
  const byModelId = new Map<string, ModelsDevModel>();
  const withPriority = preferredModels.map((model) => ({
    model,
    priority: model.provider_id === PRIMARY_PROVIDER_FILTER ? 0 : 1,
  }));
  withPriority.sort((left, right) => left.priority - right.priority);
  for (const { model } of withPriority) {
    byModelId.set(model.model_id, byModelId.get(model.model_id) ?? model);
  }
  return [...byModelId.values()];
}

function buildModelsDevById(
  modelsDevModels: ModelsDevModel[],
): Map<string, ModelsDevModel> {
  return new Map(modelsDevModels.map((model) => [model.model_id, model]));
}

function buildScrapedByModelId(
  scrapedModels: unknown[],
): Map<string, ScrapedEvalModel> {
  const scrapedByModelId = new Map<string, ScrapedEvalModel>();
  for (const model of scrapedModels) {
    const typedModel = model as ScrapedEvalModel;
    if (
      typeof typedModel.model_id === "string" &&
      typedModel.model_id.length > 0
    ) {
      scrapedByModelId.set(typedModel.model_id, typedModel);
    }
  }
  return scrapedByModelId;
}

function buildUnionFromApiMatch(
  evalModel: ArtificialAnalysisModel,
  bestMatchModelId: string,
  modelsDevById: Map<string, ModelsDevModel>,
  scrapedByModelId: Map<string, ScrapedEvalModel>,
): Record<string, unknown> {
  const modelId = artificialAnalysisModelId(evalModel);
  const scrapedModel = modelId ? scrapedByModelId.get(modelId) : undefined;
  const apiEvaluations = asRecord(evalModel.evaluations);
  const scrapedEvaluations = asRecord(scrapedModel?.evaluations);
  const mergedEvaluations = {
    ...apiEvaluations,
    ...scrapedEvaluations,
  };
  const intelligence = asRecord(scrapedModel?.intelligence);
  const intelligenceIndexCost = asRecord(scrapedModel?.intelligence_index_cost);
  const logo =
    typeof scrapedModel?.logo === "string" ? scrapedModel.logo : null;
  const matchedModelsDev = modelsDevById.get(bestMatchModelId) ?? null;
  const matchedModelFields = asRecord(matchedModelsDev?.model);
  const canonicalId = canonicalModelId(
    matchedModelsDev?.model?.id ?? bestMatchModelId,
    matchedModelsDev?.provider_id,
    matchedModelsDev?.model_id,
  );
  const {
    id: _matchedId,
    name: _matchedName,
    family: matchedFamily,
    model_id: _matchedModelId,
    slug: _matchedSlug,
    ...matchedModelFieldsWithoutIdFamilyAndModelRefs
  } = matchedModelFields;
  const evalModelFields = asRecord(evalModel);
  const {
    name: _evalName,
    slug: _evalSlug,
    evaluations: _evalEvaluations,
    model_creator: _evalModelCreator,
    scores: _evalScores,
    percentiles: _evalPercentiles,
    ...evalModelFieldsWithoutIdentity
  } = evalModelFields;

  return {
    id: canonicalId,
    provider_id: matchedModelsDev?.provider_id ?? null,
    ...matchedModelFieldsWithoutIdFamilyAndModelRefs,
    ...evalModelFieldsWithoutIdentity,
    openrouter_id: matchedModelsDev?.model?.id ?? null,
    name:
      typeof matchedModelsDev?.model?.name === "string"
        ? matchedModelsDev.model.name
        : typeof evalModel.name === "string"
          ? evalModel.name
          : null,
    aa_id: modelId,
    aa_slug: typeof evalModel.slug === "string" ? evalModel.slug : null,
    family: matchedFamily,
    logo,
    evaluations: mergedEvaluations,
    intelligence,
    intelligence_index_cost: intelligenceIndexCost,
  };
}

function buildUnionFromScrapedMatch(
  scrapedRow: ScrapedEvalModel,
  bestMatchModelId: string,
  modelsDevById: Map<string, ModelsDevModel>,
): Record<string, unknown> {
  const modelId =
    typeof scrapedRow.model_id === "string" ? scrapedRow.model_id : null;
  const slug = modelSlugFromModelId(modelId);
  const evaluations = asRecord(scrapedRow.evaluations);
  const intelligence = asRecord(scrapedRow.intelligence);
  const intelligenceIndexCost = asRecord(scrapedRow.intelligence_index_cost);
  const logo = typeof scrapedRow.logo === "string" ? scrapedRow.logo : null;
  const matchedModelsDev = modelsDevById.get(bestMatchModelId) ?? null;
  const matchedModelFields = asRecord(matchedModelsDev?.model);
  const canonicalId = canonicalModelId(
    matchedModelsDev?.model?.id ?? bestMatchModelId,
    matchedModelsDev?.provider_id,
    matchedModelsDev?.model_id,
  );
  const {
    id: _matchedId,
    name: _matchedName,
    family: matchedFamily,
    model_id: _matchedModelId,
    slug: _matchedSlug,
    ...matchedModelFieldsWithoutIdFamilyAndModelRefs
  } = matchedModelFields;

  return {
    id: canonicalId,
    provider_id: matchedModelsDev?.provider_id ?? null,
    openrouter_id: matchedModelsDev?.model?.id ?? null,
    name:
      typeof matchedModelsDev?.model?.name === "string"
        ? matchedModelsDev.model.name
        : modelId,
    aa_id: modelId,
    aa_slug: slug,
    family: matchedFamily,
    logo,
    ...matchedModelFieldsWithoutIdFamilyAndModelRefs,
    evaluations,
    intelligence,
    intelligence_index_cost: intelligenceIndexCost,
  };
}

async function buildUnionModelsFromPrimaryPathWithApiKey(
  apiKey?: string,
): Promise<Record<string, unknown>[]> {
  const [
    artificialAnalysisStats,
    artificialAnalysisScrapedStats,
    modelsDevStats,
  ] = await Promise.all([
    getArtificialAnalysisStats(apiKey ? { apiKey } : {}),
    getArtificialAnalysisScrapedEvalsOnlyStats(),
    getModelsDevStats(),
  ]);

  if (artificialAnalysisStats.models.length === 0) {
    return [];
  }

  const scopedModelsDevModels = scopeToPreferredProviderModels(
    modelsDevStats.models,
  );
  const matchMapping = await getMatchModelMapping({
    artificialAnalysisModels: artificialAnalysisStats.models,
    modelsDevModels: scopedModelsDevModels,
  });
  const modelsDevById = buildModelsDevById(scopedModelsDevModels);
  const scrapedByModelId = buildScrapedByModelId(
    artificialAnalysisScrapedStats.data,
  );
  const matchBySlug = new Map(
    matchMapping.models.map((model) => [model.artificial_analysis_slug, model]),
  );

  return artificialAnalysisStats.models
    .map((evalModel) => {
      const slug = typeof evalModel.slug === "string" ? evalModel.slug : "";
      const match = matchBySlug.get(slug);
      const bestMatchModelId = match?.best_match?.model_id;
      if (
        typeof bestMatchModelId !== "string" ||
        bestMatchModelId.length === 0
      ) {
        return null;
      }
      if (hasVariantConflict(slug, bestMatchModelId)) {
        return null;
      }
      return buildUnionFromApiMatch(
        evalModel,
        bestMatchModelId,
        modelsDevById,
        scrapedByModelId,
      );
    })
    .filter((row): row is Record<string, unknown> => row != null);
}

async function buildUnionModelsFromFallbackPath(): Promise<
  Record<string, unknown>[]
> {
  const [scrapedStats, modelsDevStats] = await Promise.all([
    getArtificialAnalysisScrapedEvalsOnlyStats(),
    getModelsDevStats(),
  ]);
  const scopedModelsDevModels = scopeToPreferredProviderModels(
    modelsDevStats.models,
  );
  const fallbackDiagnostics = await getScraperFallbackMatchDiagnostics({
    scrapedRows: scrapedStats.data,
    modelsDevModels: scopedModelsDevModels,
  });
  const modelsDevById = buildModelsDevById(scopedModelsDevModels);
  const scrapedBySlug = new Map<string, ScrapedEvalModel>();
  for (const row of scrapedStats.data) {
    const typedRow = row as ScrapedEvalModel;
    const slug = modelSlugFromModelId(typedRow.model_id);
    if (slug) {
      scrapedBySlug.set(slug, typedRow);
    }
  }

  return fallbackDiagnostics.models
    .map((model) => {
      const bestMatchModelId = model.best_match?.model_id;
      if (
        typeof bestMatchModelId !== "string" ||
        bestMatchModelId.length === 0
      ) {
        return null;
      }
      if (
        hasVariantConflict(model.artificial_analysis_slug, bestMatchModelId)
      ) {
        return null;
      }
      const scrapedRow = scrapedBySlug.get(model.artificial_analysis_slug);
      if (!scrapedRow) {
        return null;
      }
      return buildUnionFromScrapedMatch(
        scrapedRow,
        bestMatchModelId,
        modelsDevById,
      );
    })
    .filter((row): row is Record<string, unknown> => row != null);
}

async function buildUnionModelsWithApiKey(
  apiKey?: string,
): Promise<Record<string, unknown>[]> {
  // Temporary override: disable AA API source and rely on scraper fallback only.
  void apiKey;
  return buildUnionModelsFromFallbackPath();
}

function buildLogo(model: JsonObject, provider: string | null): string {
  const modelCreator = asRecord(model.model_creator);
  const logoSlug = modelCreator.slug;
  if (typeof logoSlug === "string" && logoSlug.length > 0) {
    return `https://artificialanalysis.ai/img/logos/${logoSlug}_small.svg`;
  }
  return `https://models.dev/logos/${provider ?? "unknown"}.svg`;
}

function normalizeOpenRouterSpeed(performance: unknown): JsonObject {
  const parsed = asRecord(performance);
  return {
    throughput_tokens_per_second_median: asFiniteNumber(
      parsed.throughput_tokens_per_second_median,
    ),
    latency_seconds_median: asFiniteNumber(parsed.latency_seconds_median),
    e2e_latency_seconds_median: asFiniteNumber(
      parsed.e2e_latency_seconds_median,
    ),
  };
}

function normalizeOpenRouterPricing(pricing: unknown): JsonObject {
  const parsed = asRecord(pricing);
  return {
    weighted_input: asFiniteNumber(parsed.weighted_input_price_per_1m),
    weighted_output: asFiniteNumber(parsed.weighted_output_price_per_1m),
  };
}

async function buildOpenRouterDataById(
  unionModels: Record<string, unknown>[],
): Promise<{
  speedById: Map<string, JsonObject>;
  pricingById: Map<string, JsonObject>;
}> {
  const modelIds = unionModels
    .map((row) => asRecord(row).id)
    .filter((id): id is string => typeof id === "string" && id.length > 0);
  if (modelIds.length === 0) {
    return {
      speedById: new Map(),
      pricingById: new Map(),
    };
  }

  try {
    const payload = await getOpenRouterScrapedStats({
      modelIds,
      concurrency: OPENROUTER_SPEED_CONCURRENCY,
    });
    const speedById = new Map(
      payload.models.map((model) => [
        model.id,
        normalizeOpenRouterSpeed(model.performance),
      ]),
    );
    const pricingById = new Map(
      payload.models.map((model) => [
        model.id,
        normalizeOpenRouterPricing(model.pricing),
      ]),
    );
    return { speedById, pricingById };
  } catch {
    return {
      speedById: new Map(),
      pricingById: new Map(),
    };
  }
}

function buildSpeed(
  modelId: string | null,
  openRouterSpeedById: Map<string, JsonObject>,
): JsonObject {
  if (!modelId) {
    return { ...EMPTY_OPENROUTER_SPEED };
  }
  const speed = openRouterSpeedById.get(modelId);
  if (!speed) {
    return { ...EMPTY_OPENROUTER_SPEED };
  }
  return speed;
}

function buildCost(model: JsonObject, openRouterPricing: JsonObject): unknown {
  const baseCost = asRecord(model.cost);
  const cleanedCost: JsonObject = Object.fromEntries(
    Object.entries(baseCost).filter(([, value]) => value != null),
  );
  const weightedInput = asFiniteNumber(openRouterPricing.weighted_input);
  const weightedOutput = asFiniteNumber(openRouterPricing.weighted_output);
  if (weightedInput != null) {
    cleanedCost.weighted_input = weightedInput;
  }
  if (weightedOutput != null) {
    cleanedCost.weighted_output = weightedOutput;
  }
  const blendedPrice = blendedPriceValue(model);
  if (blendedPrice != null) {
    cleanedCost.blended_price = blendedPrice;
  }
  return Object.keys(cleanedCost).length > 0 ? cleanedCost : null;
}

function intelligencePercentileValue(
  model: ModelStatsSelectedModel,
): number | null {
  return asFiniteNumber(asRecord(model.percentiles).intelligence_percentile);
}

async function writeJson(path: string, payload: unknown): Promise<void> {
  await mkdir(CACHE_DIR, { recursive: true });
  await writeFile(path, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
}

function nowEpochSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

function isFreshCache(fetchedAtEpochSeconds: unknown): boolean {
  if (typeof fetchedAtEpochSeconds !== "number") {
    return false;
  }
  const ageSeconds = nowEpochSeconds() - fetchedAtEpochSeconds;
  return ageSeconds >= 0 && ageSeconds <= CACHE_TTL_SECONDS;
}

function filterModelsById(
  models: ModelStatsSelectedModel[],
  id: string | null | undefined,
): ModelStatsSelectedModel[] {
  if (id == null) {
    return models;
  }
  return models.filter((model) => model.id === id);
}

function sortModelsByIntelligencePercentile(
  models: ModelStatsSelectedModel[],
): ModelStatsSelectedModel[] {
  return [...models].sort((left, right) => {
    const leftIntelligence = intelligencePercentileValue(left);
    const rightIntelligence = intelligencePercentileValue(right);
    if (leftIntelligence == null && rightIntelligence == null) {
      return (left.id ?? "").localeCompare(right.id ?? "");
    }
    if (leftIntelligence == null) {
      return 1;
    }
    if (rightIntelligence == null) {
      return -1;
    }
    if (leftIntelligence !== rightIntelligence) {
      return rightIntelligence - leftIntelligence;
    }
    return (left.id ?? "").localeCompare(right.id ?? "");
  });
}

function hasIntelligenceCost(row: JsonObject): boolean {
  const intelligenceIndexCost = asRecord(row.intelligence_index_cost);
  return asFiniteNumber(intelligenceIndexCost.total_cost) != null;
}

function hasScoreSignal(row: JsonObject): boolean {
  const scores = asRecord(row.scores);
  return (
    asFiniteNumber(scores.intelligence_score) != null ||
    asFiniteNumber(scores.agentic_score) != null ||
    asFiniteNumber(scores.speed_score) != null ||
    asFiniteNumber(scores.price_score) != null
  );
}

function unionRowPriority(row: JsonObject): number {
  const providerId = row.provider_id;
  const openrouterBoost =
    providerId === PRIMARY_PROVIDER_FILTER ? 1_000_000 : 0;
  const intelligenceCostBoost = hasIntelligenceCost(row) ? 1_000 : 0;
  const scoreSignalBoost = hasScoreSignal(row) ? 10 : 0;
  return openrouterBoost + intelligenceCostBoost + scoreSignalBoost;
}

function dedupeUnionModelsPreferOpenrouter(
  unionModels: Record<string, unknown>[],
): Record<string, unknown>[] {
  const groupedByNormalizedId = new Map<string, JsonObject[]>();
  const passthrough: Record<string, unknown>[] = [];

  for (const row of unionModels) {
    const typedRow = asRecord(row);
    const id = typeof typedRow.id === "string" ? typedRow.id : null;
    if (!id) {
      passthrough.push(row);
      continue;
    }
    const key = normalizeProviderModelId(id);
    const group = groupedByNormalizedId.get(key) ?? [];
    group.push(typedRow);
    groupedByNormalizedId.set(key, group);
  }

  const dedupedRows: JsonObject[] = [];
  for (const group of groupedByNormalizedId.values()) {
    const winner = [...group].sort(
      (left, right) => unionRowPriority(right) - unionRowPriority(left),
    )[0] as JsonObject;
    const mergedIntelligenceIndexCost: JsonObject = {
      ...asRecord(winner.intelligence_index_cost),
    };
    for (const candidate of group) {
      const candidateCost = asRecord(candidate.intelligence_index_cost);
      for (const [key, value] of Object.entries(candidateCost)) {
        if (mergedIntelligenceIndexCost[key] == null && value != null) {
          mergedIntelligenceIndexCost[key] = value;
        }
      }
    }
    dedupedRows.push({
      ...winner,
      intelligence_index_cost: mergedIntelligenceIndexCost,
    });
  }

  return [...passthrough, ...dedupedRows];
}

function nonFreeModelId(modelId: string): string | null {
  return modelId.endsWith(":free") ? modelId.slice(0, -":free".length) : null;
}

function hasPositiveCostFields(cost: JsonObject): boolean {
  const input = asFiniteNumber(cost.input);
  const output = asFiniteNumber(cost.output);
  return input != null && input > 0 && output != null && output > 0;
}

function backfillFreeRouteCosts(
  unionModels: Record<string, unknown>[],
): Record<string, unknown>[] {
  const nonFreeCostById = new Map<string, JsonObject>();
  for (const row of unionModels) {
    const rowRecord = asRecord(row);
    const id = typeof rowRecord.id === "string" ? rowRecord.id : null;
    if (!id || id.endsWith(":free")) {
      continue;
    }
    const cost = asRecord(rowRecord.cost);
    if (hasPositiveCostFields(cost)) {
      nonFreeCostById.set(id, cost);
    }
  }

  return unionModels.map((row) => {
    const rowRecord = asRecord(row);
    const id = typeof rowRecord.id === "string" ? rowRecord.id : null;
    if (!id) {
      return row;
    }
    const baseId = nonFreeModelId(id);
    if (!baseId) {
      return row;
    }
    const baseCost = nonFreeCostById.get(baseId);
    if (!baseCost) {
      return row;
    }
    return {
      ...rowRecord,
      cost: {
        ...baseCost,
      },
    };
  });
}

function isPlainObject(value: unknown): value is JsonObject {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function isWithinRecentLookback(
  releaseDate: string | null,
  lookbackDays: number,
): boolean {
  if (typeof releaseDate !== "string" || releaseDate.length === 0) {
    return false;
  }
  const releaseTimestampMs = Date.parse(releaseDate);
  if (!Number.isFinite(releaseTimestampMs)) {
    return false;
  }
  const cutoffMs = Date.now() - lookbackDays * 24 * 60 * 60 * 1000;
  return releaseTimestampMs >= cutoffMs;
}

function selectPruneSampleModels(
  models: ModelStatsSelectedModel[],
): ModelStatsSelectedModel[] {
  const recentModels = models.filter((model) =>
    isWithinRecentLookback(
      model.release_date,
      NULL_FIELD_PRUNE_RECENT_LOOKBACK_DAYS,
    ),
  );
  return recentModels.length > 0 ? recentModels : models;
}

function countNullishTopLevelKey(
  models: ModelStatsSelectedModel[],
  key: string,
): number {
  return models.reduce((count, model) => {
    const modelRecord = asRecord(model);
    return modelRecord[key] == null ? count + 1 : count;
  }, 0);
}

function countNullishNestedKey(
  models: ModelStatsSelectedModel[],
  parentKey: string,
  nestedKey: string,
): number {
  return models.reduce((count, model) => {
    const modelRecord = asRecord(model);
    const parentValue = modelRecord[parentKey];
    if (!isPlainObject(parentValue) || parentValue[nestedKey] == null) {
      return count + 1;
    }
    return count;
  }, 0);
}

function pruneSparseFields(
  models: ModelStatsSelectedModel[],
  nullThreshold: number = NULL_FIELD_PRUNE_THRESHOLD,
): ModelStatsSelectedModel[] {
  if (models.length === 0) {
    return models;
  }

  const sampleModels = selectPruneSampleModels(models);
  const sampleTotal = sampleModels.length;
  const topLevelKeys = new Set<string>();
  const nestedKeysByParent = new Map<string, Set<string>>();

  for (const model of models) {
    for (const [key, value] of Object.entries(model)) {
      topLevelKeys.add(key);
      if (!isPlainObject(value)) {
        continue;
      }
      const nestedKeys = nestedKeysByParent.get(key) ?? new Set<string>();
      for (const nestedKey of Object.keys(value)) {
        nestedKeys.add(nestedKey);
      }
      nestedKeysByParent.set(key, nestedKeys);
    }
  }

  const topLevelKeysToPrune = new Set<string>();
  for (const key of topLevelKeys) {
    if (STABLE_TOP_LEVEL_KEYS.has(key)) {
      continue;
    }
    const nullCount = countNullishTopLevelKey(sampleModels, key);
    if (nullCount / sampleTotal > nullThreshold) {
      topLevelKeysToPrune.add(key);
    }
  }

  const nestedKeysToPruneByParent = new Map<string, Set<string>>();
  for (const [parentKey, nestedKeys] of nestedKeysByParent) {
    if (parentKey !== "evaluations") {
      continue;
    }
    const keysToPrune = new Set<string>();
    for (const nestedKey of nestedKeys) {
      const nullCount = countNullishNestedKey(
        sampleModels,
        parentKey,
        nestedKey,
      );
      if (nullCount / sampleTotal > nullThreshold) {
        keysToPrune.add(nestedKey);
      }
    }
    if (keysToPrune.size > 0) {
      nestedKeysToPruneByParent.set(parentKey, keysToPrune);
    }
  }

  const prunedModels = models.map((model) => {
    const nextModel: JsonObject = { ...model };
    for (const key of topLevelKeysToPrune) {
      delete nextModel[key];
    }
    for (const [parentKey, nestedKeysToPrune] of nestedKeysToPruneByParent) {
      const parentValue = nextModel[parentKey];
      if (!isPlainObject(parentValue)) {
        continue;
      }
      const nextParentValue: JsonObject = { ...parentValue };
      for (const nestedKey of nestedKeysToPrune) {
        delete nextParentValue[nestedKey];
      }
      nextModel[parentKey] = nextParentValue;
    }
    return nextModel;
  });
  return prunedModels as ModelStatsSelectedModel[];
}

/**
 * Persist the final model stats payload to disk.
 *
 * Write failures are intentionally swallowed to keep API behavior in-memory
 * first.
 */
export async function saveModelStatsSelected(
  payload: ModelStatsSelectedPayload,
  outputPath = DEFAULT_OUTPUT_PATH,
): Promise<void> {
  try {
    await writeJson(outputPath, payload);
  } catch {
    // Intentionally swallow cache write errors: API remains in-memory first.
  }
}

async function loadModelStatsSelectedFromCache(
  outputPath: string,
): Promise<ModelStatsSelectedPayload | null> {
  try {
    const content = await readFile(outputPath, "utf-8");
    const payload = JSON.parse(content) as ModelStatsSelectedPayload;
    if (!Array.isArray(payload.models)) {
      return null;
    }
    if (!isFreshCache(payload.fetched_at_epoch_seconds)) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}

function mapUnionModelToSelected(
  unionModel: unknown,
  openRouterSpeedById: Map<string, JsonObject>,
  openRouterPricingById: Map<string, JsonObject>,
  speedOutputTokenAnchors: number[],
): ModelStatsSelectedModel {
  const model = asRecord(unionModel);
  const provider = providerFromModel(model);
  const modelId = typeof model.id === "string" ? model.id : null;
  const speed = buildSpeed(modelId, openRouterSpeedById);
  const pricing =
    (modelId != null ? openRouterPricingById.get(modelId) : null) ??
    EMPTY_OPENROUTER_PRICING;
  return {
    id: modelId,
    name: typeof model.name === "string" ? model.name : null,
    provider,
    logo: buildLogo(model, provider),
    attachment: typeof model.attachment === "boolean" ? model.attachment : null,
    reasoning: typeof model.reasoning === "boolean" ? model.reasoning : null,
    release_date:
      typeof model.release_date === "string" ? model.release_date : null,
    modalities: model.modalities ?? null,
    open_weights:
      typeof model.open_weights === "boolean" ? model.open_weights : null,
    cost: buildCost(model, pricing),
    context_window: model.limit ?? null,
    speed,
    intelligence: buildIntelligence(model),
    intelligence_index_cost: buildIntelligenceIndexCost(model),
    evaluations: buildEvaluations(model),
    scores: buildScores(model, speed, speedOutputTokenAnchors),
    percentiles: null,
  };
}

/**
 * Return final model stats enriched from source data + matcher links.
 *
 * Design:
 * - list mode (`id == null`): cache-first (< 1 day), else recompute and save
 * - single-model mode (`id != null`): in-memory only, exact-id filtering
 * - failure mode: never throw; returns `{ fetched_at_epoch_seconds: null, models: [] }`
 */
export async function getModelStatsSelected(
  options: ModelStatsSelectedOptions = {},
): Promise<ModelStatsSelectedPayload> {
  try {
    if (options.id == null) {
      const cachedPayload =
        await loadModelStatsSelectedFromCache(DEFAULT_OUTPUT_PATH);
      if (cachedPayload) {
        return cachedPayload;
      }
    }

    const unionModels = await buildUnionModelsWithApiKey(options.apiKey);
    const dedupedUnionModels = dedupeUnionModelsPreferOpenrouter(unionModels);
    const unionModelsWithCostBackfill =
      backfillFreeRouteCosts(dedupedUnionModels);
    const {
      speedById: openRouterSpeedById,
      pricingById: openRouterPricingById,
    } = await buildOpenRouterDataById(unionModelsWithCostBackfill);
    const speedOutputTokenAnchors =
      deriveSpeedOutputTokenAnchors(openRouterSpeedById);
    const allModels = unionModelsWithCostBackfill.map((model) =>
      mapUnionModelToSelected(
        model,
        openRouterSpeedById,
        openRouterPricingById,
        speedOutputTokenAnchors,
      ),
    );
    const modelsWithComputedPercentiles = withComputedPercentiles(allModels);
    const sortedModels = sortModelsByIntelligencePercentile(
      modelsWithComputedPercentiles,
    );
    const prunedModels = pruneSparseFields(sortedModels);
    const filteredModels = filterModelsById(prunedModels, options.id);
    const fetchedAt = nowEpochSeconds();

    if (options.id != null) {
      return {
        fetched_at_epoch_seconds: fetchedAt,
        models: filteredModels,
      };
    }

    const listPayload: ModelStatsSelectedPayload = {
      fetched_at_epoch_seconds: fetchedAt,
      models: filteredModels,
    };
    await saveModelStatsSelected(listPayload, DEFAULT_OUTPUT_PATH);
    return listPayload;
  } catch {
    return {
      fetched_at_epoch_seconds: null,
      models: [],
    };
  }
}
