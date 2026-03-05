import {
  PRIMARY_PROVIDER_ID,
  asFiniteNumber,
  asRecord,
  normalizeProviderModelId,
  type JsonObject,
} from "../shared.js";

import { type ModelStatsSelectedModel } from "./types.js";

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

function intelligencePercentileValue(
  model: ModelStatsSelectedModel,
): number | null {
  return asFiniteNumber(asRecord(model.percentiles).intelligence_percentile);
}

export function filterModelsById(
  models: ModelStatsSelectedModel[],
  id: string | null | undefined,
): ModelStatsSelectedModel[] {
  if (id == null) {
    return models;
  }
  return models.filter((model) => model.id === id);
}

export function sortModelsByIntelligencePercentile(
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

function hasIntelligenceCost(unionRow: JsonObject): boolean {
  const intelligenceIndexCost = asRecord(unionRow.intelligence_index_cost);
  return asFiniteNumber(intelligenceIndexCost.total_cost) != null;
}

function hasScoreSignal(unionRow: JsonObject): boolean {
  const scores = asRecord(unionRow.scores);
  return (
    asFiniteNumber(scores.intelligence_score) != null ||
    asFiniteNumber(scores.agentic_score) != null ||
    asFiniteNumber(scores.speed_score) != null ||
    asFiniteNumber(scores.price_score) != null
  );
}

function unionRowPriority(unionRow: JsonObject): number {
  const providerId = unionRow.provider_id;
  const openrouterBoost = providerId === PRIMARY_PROVIDER_ID ? 1_000_000 : 0;
  const intelligenceCostBoost = hasIntelligenceCost(unionRow) ? 1_000 : 0;
  const scoreSignalBoost = hasScoreSignal(unionRow) ? 10 : 0;
  return openrouterBoost + intelligenceCostBoost + scoreSignalBoost;
}

export function dedupeUnionModelsPreferOpenrouter(
  unionModels: Record<string, unknown>[],
): Record<string, unknown>[] {
  const groupedByNormalizedId = new Map<string, JsonObject[]>();
  const passthrough: Record<string, unknown>[] = [];

  for (const unionModel of unionModels) {
    const unionRow = asRecord(unionModel);
    const id = typeof unionRow.id === "string" ? unionRow.id : null;
    if (!id) {
      passthrough.push(unionModel);
      continue;
    }
    const key = normalizeProviderModelId(id);
    const group = groupedByNormalizedId.get(key) ?? [];
    group.push(unionRow);
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

export function backfillFreeRouteCosts(
  unionModels: Record<string, unknown>[],
): Record<string, unknown>[] {
  const nonFreeCostById = new Map<string, JsonObject>();
  for (const unionModel of unionModels) {
    const unionRow = asRecord(unionModel);
    const id = typeof unionRow.id === "string" ? unionRow.id : null;
    if (!id || id.endsWith(":free")) {
      continue;
    }
    const cost = asRecord(unionRow.cost);
    if (hasPositiveCostFields(cost)) {
      nonFreeCostById.set(id, cost);
    }
  }

  return unionModels.map((unionModel) => {
    const unionRow = asRecord(unionModel);
    const id = typeof unionRow.id === "string" ? unionRow.id : null;
    if (!id) {
      return unionModel;
    }
    const baseId = nonFreeModelId(id);
    if (!baseId) {
      return unionModel;
    }
    const baseCost = nonFreeCostById.get(baseId);
    if (!baseCost) {
      return unionModel;
    }
    return {
      ...unionRow,
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

export function pruneSparseFields(
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
