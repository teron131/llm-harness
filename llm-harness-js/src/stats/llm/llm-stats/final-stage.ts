import { asFiniteNumber, asRecord, type JsonObject } from "../shared.js";

import {
  attachPercentiles,
  blendedPriceValue,
  buildScores,
} from "./scoring.js";
import { type EnrichedRows, type ModelStatsSelectedModel } from "./types.js";

const EMPTY_OPENROUTER_PRICING = {
  weighted_input: null,
  weighted_output: null,
} as const;
const MIN_INTELLIGENCE_COST_TOKEN_THRESHOLD = 1_000_000;
const INTELLIGENCE_COST_TOTAL_COST_KEY = "intelligence_index_cost_total_cost";
const INTELLIGENCE_COST_TOTAL_TOKENS_KEY =
  "intelligence_index_cost_total_tokens";
const NULL_FIELD_PRUNE_THRESHOLD = 0.5;
const NULL_FIELD_PRUNE_RECENT_LOOKBACK_DAYS = 90;
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

function providerFromModel(model: JsonObject): string | null {
  const fromId = providerFromId(model.id);
  if (fromId) {
    return fromId;
  }
  return typeof model.provider_id === "string" ? model.provider_id : null;
}

function buildLogo(model: JsonObject, provider: string | null): string {
  const modelCreator = asRecord(model.model_creator);
  const logoSlug = modelCreator.slug;
  if (typeof logoSlug === "string" && logoSlug.length > 0) {
    return `https://artificialanalysis.ai/img/logos/${logoSlug}_small.svg`;
  }
  return `https://models.dev/logos/${provider ?? "unknown"}.svg`;
}

function buildSpeed(
  model: JsonObject,
  modelId: string | null,
  openRouterSpeedById: Map<string, JsonObject>,
): JsonObject {
  const openRouterSpeed = modelId ? openRouterSpeedById.get(modelId) : null;
  const throughput =
    asFiniteNumber(openRouterSpeed?.throughput_tokens_per_second_median) ??
    asFiniteNumber(model.median_output_tokens_per_second);
  const latency =
    asFiniteNumber(openRouterSpeed?.latency_seconds_median) ??
    asFiniteNumber(model.median_time_to_first_token_seconds);
  const e2eLatency =
    asFiniteNumber(openRouterSpeed?.e2e_latency_seconds_median) ??
    asFiniteNumber(model.median_time_to_first_answer_token) ??
    latency;
  return {
    throughput_tokens_per_second_median: throughput,
    latency_seconds_median: latency,
    e2e_latency_seconds_median: e2eLatency,
  };
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
  const blendedPrice = blendedPriceValue(cleanedCost);
  if (blendedPrice != null) {
    cleanedCost.blended_price = blendedPrice;
  }
  return Object.keys(cleanedCost).length > 0 ? cleanedCost : null;
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

function intelligencePercentileValue(
  model: ModelStatsSelectedModel,
): number | null {
  return asFiniteNumber(asRecord(model.percentiles).intelligence_percentile);
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

  return models.map((model) => {
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
    return nextModel as ModelStatsSelectedModel;
  });
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

function projectFinalModel(
  row: unknown,
  openRouterSpeedById: Map<string, JsonObject>,
  openRouterPricingById: Map<string, JsonObject>,
  speedOutputTokenAnchors: number[],
): ModelStatsSelectedModel {
  const model = asRecord(row);
  const provider = providerFromModel(model);
  const modelId = typeof model.id === "string" ? model.id : null;
  const speed = buildSpeed(model, modelId, openRouterSpeedById);
  const pricing =
    (modelId != null ? openRouterPricingById.get(modelId) : null) ??
    EMPTY_OPENROUTER_PRICING;
  const cost = buildCost(model, pricing);
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
    cost,
    context_window: model.limit ?? null,
    speed,
    intelligence: buildIntelligence(model),
    intelligence_index_cost: buildIntelligenceIndexCost(model),
    evaluations: buildEvaluations(model),
    scores: buildScores(model, cost, speed, speedOutputTokenAnchors),
    percentiles: null,
  };
}

export function buildFinalModels(
  enrichedRows: EnrichedRows,
  id: string | null | undefined,
): ModelStatsSelectedModel[] {
  const models = enrichedRows.rows.map((row) =>
    projectFinalModel(
      row,
      enrichedRows.openRouterSpeedById,
      enrichedRows.openRouterPricingById,
      enrichedRows.speedOutputTokenAnchors,
    ),
  );
  const modelsWithPercentiles = attachPercentiles(models);
  const sortedModels = sortModelsByIntelligencePercentile(
    modelsWithPercentiles,
  );
  const prunedModels = pruneSparseFields(sortedModels);
  return filterModelsById(prunedModels, id);
}
