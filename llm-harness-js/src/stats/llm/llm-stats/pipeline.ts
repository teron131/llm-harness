import { getOpenRouterScrapedStats } from "../sources/openrouter-scraper.js";
import { asFiniteNumber, asRecord, type JsonObject } from "../shared.js";

import {
  backfillFreeModelCosts,
  dedupeRowsPreferOpenRouter,
  filterModelsById,
  pruneSparseFields,
  sortModelsByIntelligencePercentile,
} from "./postprocess.js";
import {
  blendedPriceValue,
  buildEvaluations,
  buildIntelligence,
  buildIntelligenceIndexCost,
  buildScores,
  deriveSpeedOutputTokenAnchors,
  attachPercentiles,
} from "./scoring.js";
import {
  type EnrichedUnionRows,
  type ModelStatsSelectedModel,
} from "./types.js";

const OPENROUTER_SPEED_CONCURRENCY = 8;
const EMPTY_OPENROUTER_PRICING = {
  weighted_input: null,
  weighted_output: null,
} as const;
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
  rows: Record<string, unknown>[],
): Promise<{
  speedById: Map<string, JsonObject>;
  pricingById: Map<string, JsonObject>;
}> {
  const modelIds = rows
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

function mapUnionModelToSelected(
  unionModel: unknown,
  openRouterSpeedById: Map<string, JsonObject>,
  openRouterPricingById: Map<string, JsonObject>,
  speedOutputTokenAnchors: number[],
): ModelStatsSelectedModel {
  const model = asRecord(unionModel);
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

export async function enrichRows(
  matchedRows: Record<string, unknown>[],
): Promise<EnrichedUnionRows> {
  const dedupedRows = dedupeRowsPreferOpenRouter(matchedRows);
  const rows = backfillFreeModelCosts(dedupedRows);
  const { speedById: openRouterSpeedById, pricingById: openRouterPricingById } =
    await buildOpenRouterDataById(rows);
  const speedOutputTokenAnchors =
    deriveSpeedOutputTokenAnchors(openRouterSpeedById);
  return {
    unionRows: rows,
    openRouterSpeedById,
    openRouterPricingById,
    speedOutputTokenAnchors,
  };
}

export function buildFinalModels(
  enrichedRows: EnrichedUnionRows,
  id: string | null | undefined,
): ModelStatsSelectedModel[] {
  const models = enrichedRows.unionRows.map((row) =>
    mapUnionModelToSelected(
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
