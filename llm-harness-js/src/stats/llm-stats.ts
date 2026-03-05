import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { getArtificialAnalysisStats } from "./data-sources/llm/artificial-analysis-api";
import { getArtificialAnalysisScrapedEvalsOnlyStats } from "./data-sources/llm/artificial-analysis-scraper";
import {
  getMatchModelMapping,
  getScraperFallbackMatchDiagnostics,
} from "./data-sources/llm/matcher";
import { getModelsDevStats } from "./data-sources/llm/models-dev";
import { percentileRank } from "./data-sources/utils";

const DEFAULT_OUTPUT_PATH = resolve(".cache/model_stats.json");
const CACHE_DIR = resolve(".cache");
const CACHE_TTL_SECONDS = 60 * 60 * 24;
const NULL_FIELD_PRUNE_THRESHOLD = 0.5;
const NULL_FIELD_PRUNE_RECENT_LOOKBACK_DAYS = 90;
const MIN_INTELLIGENCE_COST_TOKEN_THRESHOLD = 1_000_000;
const INTELLIGENCE_COST_TOTAL_COST_KEY = "intelligence_index_cost_total_cost";
const INTELLIGENCE_COST_TOTAL_TOKENS_KEY =
  "intelligence_index_cost_total_tokens";
const INTELLIGENCE_COST_KEYS = [
  INTELLIGENCE_COST_TOTAL_COST_KEY,
  INTELLIGENCE_COST_TOTAL_TOKENS_KEY,
] as const;
const LETTER_SUFFIX_REGEX = /^[a-z-]+$/;
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
  "evaluations",
  "scores",
  "percentiles",
]);

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

function normalizeIntelligenceCostFields(intelligence: JsonObject): JsonObject {
  const totalTokens = asFiniteNumber(
    intelligence[INTELLIGENCE_COST_TOTAL_TOKENS_KEY],
  );
  const totalCost = asFiniteNumber(
    intelligence[INTELLIGENCE_COST_TOTAL_COST_KEY],
  );
  if (
    totalTokens == null ||
    totalTokens < MIN_INTELLIGENCE_COST_TOKEN_THRESHOLD ||
    totalCost == null ||
    totalCost <= 0
  ) {
    return {
      ...intelligence,
      [INTELLIGENCE_COST_TOTAL_TOKENS_KEY]: null,
      [INTELLIGENCE_COST_TOTAL_COST_KEY]: null,
    };
  }
  return {
    ...intelligence,
    [INTELLIGENCE_COST_TOTAL_TOKENS_KEY]: totalTokens,
    [INTELLIGENCE_COST_TOTAL_COST_KEY]: totalCost,
  };
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
  const normalizedIntelligence = normalizeIntelligenceCostFields(intelligence);
  return Object.keys(normalizedIntelligence).length > 0
    ? normalizedIntelligence
    : null;
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

function buildScores(model: JsonObject): unknown {
  const baseScores = asRecord(model.scores);
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
  const intelligenceScore = meanOfFinite([
    intelligenceIndex,
    intelligenceBenchmarkMean,
  ]);
  const agenticScore = meanOfFinite([agenticIndex, agenticBenchmarkMean]);
  if (
    intelligenceScore == null &&
    agenticScore == null &&
    asFiniteNumber(baseScores.overall_score) == null &&
    asFiniteNumber(baseScores.price_score) == null &&
    asFiniteNumber(baseScores.speed_score) == null
  ) {
    return null;
  }
  return {
    overall_score: asFiniteNumber(baseScores.overall_score),
    intelligence_score: intelligenceScore,
    agentic_score: agenticScore,
    price_score: asFiniteNumber(baseScores.price_score),
    speed_score: asFiniteNumber(baseScores.speed_score),
  };
}

function withAgenticPercentiles(
  models: ModelStatsSelectedModel[],
): ModelStatsSelectedModel[] {
  const agenticScores = models.map((model) =>
    asFiniteNumber(asRecord(model.scores).agentic_score),
  );

  return models.map((model) => {
    const scores = asRecord(model.scores);
    const percentiles = asRecord(model.percentiles);
    const agenticScore = asFiniteNumber(scores.agentic_score);
    const agenticPercentile =
      agenticScore == null ? null : percentileRank(agenticScores, agenticScore);
    const {
      overall_percentile: _overallPercentile,
      intelligence_percentile: _intelligencePercentile,
      agentic_percentile: _agenticPercentile,
      speed_percentile: _speedPercentile,
      price_percentile: _pricePercentile,
      ...remainingPercentiles
    } = percentiles;

    return {
      ...model,
      percentiles: {
        overall_percentile: percentiles.overall_percentile ?? null,
        intelligence_percentile: percentiles.intelligence_percentile ?? null,
        agentic_percentile: agenticPercentile,
        speed_percentile: percentiles.speed_percentile ?? null,
        price_percentile: percentiles.price_percentile ?? null,
        ...remainingPercentiles,
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
  const primaryModels = await buildUnionModelsFromPrimaryPathWithApiKey(apiKey);
  if (primaryModels.length > 0) {
    return primaryModels;
  }
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

function buildSpeed(model: JsonObject): JsonObject {
  return Object.fromEntries(
    Object.entries(model).filter(([key]) => key.startsWith("median_")),
  );
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

function hasIntelligenceCost(row: JsonObject): boolean {
  const intelligence = asRecord(row.intelligence);
  return asFiniteNumber(intelligence[INTELLIGENCE_COST_TOTAL_COST_KEY]) != null;
}

function hasOverallScore(row: JsonObject): boolean {
  const scores = asRecord(row.scores);
  return asFiniteNumber(scores.overall_score) != null;
}

function unionRowPriority(row: JsonObject): number {
  const providerId = row.provider_id;
  const openrouterBoost =
    providerId === PRIMARY_PROVIDER_FILTER ? 1_000_000 : 0;
  const intelligenceCostBoost = hasIntelligenceCost(row) ? 1_000 : 0;
  const overallScoreBoost = hasOverallScore(row) ? 10 : 0;
  return openrouterBoost + intelligenceCostBoost + overallScoreBoost;
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
    const winnerIntelligence = asRecord(winner.intelligence);
    const mergedIntelligence: JsonObject = { ...winnerIntelligence };
    for (const key of INTELLIGENCE_COST_KEYS) {
      if (asFiniteNumber(mergedIntelligence[key]) != null) {
        continue;
      }
      for (const candidate of group) {
        const candidateIntelligence = asRecord(candidate.intelligence);
        if (asFiniteNumber(candidateIntelligence[key]) != null) {
          mergedIntelligence[key] = candidateIntelligence[key];
          break;
        }
      }
    }
    dedupedRows.push({
      ...winner,
      intelligence: mergedIntelligence,
    });
  }

  return [...passthrough, ...dedupedRows];
}

async function backfillIntelligenceCostFromScraped(
  models: ModelStatsSelectedModel[],
): Promise<ModelStatsSelectedModel[]> {
  const scrapedStats = await getArtificialAnalysisScrapedEvalsOnlyStats();
  const scrapedById = new Map<string, JsonObject>();
  for (const row of scrapedStats.data) {
    const rowRecord = asRecord(row);
    const modelId = rowRecord.model_id;
    if (typeof modelId !== "string" || modelId.length === 0) {
      continue;
    }
    scrapedById.set(normalizeProviderModelId(modelId), rowRecord);
  }

  const pickBackfillValue = (
    baseFamilyId: string,
    key: (typeof INTELLIGENCE_COST_KEYS)[number],
  ): number | null => {
    const direct = scrapedById.get(baseFamilyId);
    const directValue = asFiniteNumber(asRecord(direct?.intelligence)[key]);
    if (directValue != null) {
      return directValue;
    }
    let fallbackValue: number | null = null;
    for (const [scrapedId, row] of scrapedById) {
      if (!scrapedId.startsWith(`${baseFamilyId}-`)) {
        continue;
      }
      const suffix = scrapedId.slice(baseFamilyId.length + 1);
      if (!LETTER_SUFFIX_REGEX.test(suffix)) {
        continue;
      }
      const candidateValue = asFiniteNumber(asRecord(row.intelligence)[key]);
      if (candidateValue != null) {
        fallbackValue = candidateValue;
        break;
      }
    }
    return fallbackValue;
  };

  return models.map((model) => {
    if (typeof model.id !== "string" || model.id.length === 0) {
      return model;
    }
    const baseFamilyId = normalizeProviderModelId(model.id);
    const intelligence = asRecord(model.intelligence);
    const hasRawCost = intelligence[INTELLIGENCE_COST_TOTAL_COST_KEY] != null;
    const hasRawTokens =
      intelligence[INTELLIGENCE_COST_TOTAL_TOKENS_KEY] != null;
    const cost = hasRawCost
      ? asFiniteNumber(intelligence[INTELLIGENCE_COST_TOTAL_COST_KEY])
      : pickBackfillValue(baseFamilyId, INTELLIGENCE_COST_TOTAL_COST_KEY);
    const tokens = hasRawTokens
      ? asFiniteNumber(intelligence[INTELLIGENCE_COST_TOTAL_TOKENS_KEY])
      : pickBackfillValue(baseFamilyId, INTELLIGENCE_COST_TOTAL_TOKENS_KEY);
    const normalizedIntelligence = normalizeIntelligenceCostFields({
      ...intelligence,
      [INTELLIGENCE_COST_TOTAL_COST_KEY]: cost,
      [INTELLIGENCE_COST_TOTAL_TOKENS_KEY]: tokens,
    });
    return {
      ...model,
      intelligence: normalizedIntelligence,
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

function mapUnionModelToSelected(unionModel: unknown): ModelStatsSelectedModel {
  const model = asRecord(unionModel);
  const provider = providerFromModel(model);
  return {
    id: typeof model.id === "string" ? model.id : null,
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
    cost: model.cost ?? null,
    context_window: model.limit ?? null,
    speed: buildSpeed(model),
    intelligence: buildIntelligence(model),
    evaluations: buildEvaluations(model),
    scores: buildScores(model),
    percentiles: model.percentiles ?? null,
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
    const allModels = dedupedUnionModels.map(mapUnionModelToSelected);
    const modelsWithCostBackfill =
      await backfillIntelligenceCostFromScraped(allModels);
    const modelsWithAgenticPercentiles = withAgenticPercentiles(
      modelsWithCostBackfill,
    );
    const prunedModels = pruneSparseFields(modelsWithAgenticPercentiles);
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
