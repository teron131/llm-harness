import { getArtificialAnalysisStats } from "../sources/artificial-analysis-api.js";
import { getArtificialAnalysisScrapedEvalsOnlyStats } from "../sources/artificial-analysis-scraper.js";
import { getModelsDevStats } from "../sources/models-dev.js";
import { getOpenRouterScrapedStats } from "../sources/openrouter-scraper.js";
import { getScraperFallbackMatchDiagnostics } from "../matcher.js";
import {
  FALLBACK_PROVIDER_IDS,
  PRIMARY_PROVIDER_ID,
  asFiniteNumber,
  asRecord,
  modelSlugFromModelId,
  normalizeProviderModelId,
  type JsonObject,
} from "../shared.js";

import {
  backfillFreeRouteCosts,
  dedupeUnionModelsPreferOpenrouter,
  filterModelsById,
  pruneSparseFields,
  sortModelsByIntelligencePercentile,
} from "./postprocess.js";
import {
  buildEvaluations,
  buildIntelligence,
  buildIntelligenceIndexCost,
  buildScores,
  deriveSpeedOutputTokenAnchors,
  withComputedPercentiles,
} from "./scoring.js";
import {
  type EnrichedUnionRows,
  type ModelStatsSelectedModel,
  type ModelsDevModel,
  type ScrapedEvalModel,
  type SelectedSourceData,
} from "./types.js";

const OPENROUTER_SPEED_CONCURRENCY = 8;
const EMPTY_OPENROUTER_PRICING = {
  weighted_input: null,
  weighted_output: null,
} as const;
const WEIGHTED_PRICE_INPUT_RATIO = 0.75;
const WEIGHTED_PRICE_OUTPUT_RATIO = 0.25;

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

function scopeToPreferredProviderModels(
  modelsDevModels: ModelsDevModel[],
): ModelsDevModel[] {
  const preferredModels = modelsDevModels.filter(
    (modelsDevModel) =>
      modelsDevModel.provider_id === PRIMARY_PROVIDER_ID ||
      FALLBACK_PROVIDER_IDS.has(modelsDevModel.provider_id),
  );
  const byModelId = new Map<string, ModelsDevModel>();
  const withPriority = preferredModels.map((modelsDevModel) => ({
    modelsDevModel,
    priority: modelsDevModel.provider_id === PRIMARY_PROVIDER_ID ? 0 : 1,
  }));
  withPriority.sort((left, right) => left.priority - right.priority);
  for (const { modelsDevModel } of withPriority) {
    byModelId.set(
      modelsDevModel.model_id,
      byModelId.get(modelsDevModel.model_id) ?? modelsDevModel,
    );
  }
  return [...byModelId.values()];
}

function buildModelsDevById(
  modelsDevModels: ModelsDevModel[],
): Map<string, ModelsDevModel> {
  return new Map(
    modelsDevModels.map((modelsDevModel) => [
      modelsDevModel.model_id,
      modelsDevModel,
    ]),
  );
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

function buildUnionFromScrapedMatch(
  scrapedModel: ScrapedEvalModel,
  matchedModelId: string,
  modelsDevById: Map<string, ModelsDevModel>,
): Record<string, unknown> {
  const artificialAnalysisModelId =
    typeof scrapedModel.model_id === "string" ? scrapedModel.model_id : null;
  const artificialAnalysisSlug = modelSlugFromModelId(
    artificialAnalysisModelId,
  );
  const evaluations = asRecord(scrapedModel.evaluations);
  const intelligence = asRecord(scrapedModel.intelligence);
  const intelligenceIndexCost = asRecord(scrapedModel.intelligence_index_cost);
  const logo = typeof scrapedModel.logo === "string" ? scrapedModel.logo : null;
  const matchedModelsDev = modelsDevById.get(matchedModelId) ?? null;
  const matchedModelFields = asRecord(matchedModelsDev?.model);
  const canonicalId = canonicalModelId(
    matchedModelsDev?.model?.id ?? matchedModelId,
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
        : artificialAnalysisModelId,
    aa_id: artificialAnalysisModelId,
    aa_slug: artificialAnalysisSlug,
    family: matchedFamily,
    logo,
    ...matchedModelFieldsWithoutIdFamilyAndModelRefs,
    evaluations,
    intelligence,
    intelligence_index_cost: intelligenceIndexCost,
  };
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
  const blendedPrice = blendedPriceValue(model);
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

export async function fetchSelectedSourceData(
  apiKey?: string,
): Promise<SelectedSourceData> {
  const [
    artificialAnalysisStats,
    artificialAnalysisScrapedStats,
    modelsDevStats,
  ] = await Promise.all([
    getArtificialAnalysisStats(apiKey ? { apiKey } : {}),
    getArtificialAnalysisScrapedEvalsOnlyStats(),
    getModelsDevStats(),
  ]);
  const scopedModelsDevModels = scopeToPreferredProviderModels(
    modelsDevStats.models,
  );
  const scrapedBySlug = new Map<string, ScrapedEvalModel>();
  for (const scrapedRow of artificialAnalysisScrapedStats.data) {
    const scrapedModel = scrapedRow as ScrapedEvalModel;
    const artificialAnalysisSlug = modelSlugFromModelId(scrapedModel.model_id);
    if (artificialAnalysisSlug) {
      scrapedBySlug.set(artificialAnalysisSlug, scrapedModel);
    }
  }
  return {
    artificialAnalysisModels: artificialAnalysisStats.models,
    scrapedRows: artificialAnalysisScrapedStats.data,
    scopedModelsDevModels,
    modelsDevById: buildModelsDevById(scopedModelsDevModels),
    scrapedBySlug,
  };
}

export async function buildMatchedUnionRows(
  sourceData: SelectedSourceData,
): Promise<Record<string, unknown>[]> {
  const fallbackDiagnostics = await getScraperFallbackMatchDiagnostics({
    scrapedRows: sourceData.scrapedRows,
    modelsDevModels: sourceData.scopedModelsDevModels,
  });

  return fallbackDiagnostics.models
    .map((matchedModel) => {
      const matchedModelId = matchedModel.best_match?.model_id;
      if (typeof matchedModelId !== "string" || matchedModelId.length === 0) {
        return null;
      }
      if (
        hasVariantConflict(
          matchedModel.artificial_analysis_slug,
          matchedModelId,
        )
      ) {
        return null;
      }
      const scrapedModel = sourceData.scrapedBySlug.get(
        matchedModel.artificial_analysis_slug,
      );
      if (!scrapedModel) {
        return null;
      }
      return buildUnionFromScrapedMatch(
        scrapedModel,
        matchedModelId,
        sourceData.modelsDevById,
      );
    })
    .filter(
      (unionRow): unionRow is Record<string, unknown> => unionRow != null,
    );
}

export async function enrichUnionRowsWithFallbacks(
  matchedUnionRows: Record<string, unknown>[],
): Promise<EnrichedUnionRows> {
  const dedupedUnionRows = dedupeUnionModelsPreferOpenrouter(matchedUnionRows);
  const unionRowsWithCostBackfill = backfillFreeRouteCosts(dedupedUnionRows);
  const { speedById: openRouterSpeedById, pricingById: openRouterPricingById } =
    await buildOpenRouterDataById(unionRowsWithCostBackfill);
  const speedOutputTokenAnchors =
    deriveSpeedOutputTokenAnchors(openRouterSpeedById);
  return {
    unionRows: unionRowsWithCostBackfill,
    openRouterSpeedById,
    openRouterPricingById,
    speedOutputTokenAnchors,
  };
}

export function projectSelectedRowsWithScores(
  enrichedUnionRows: EnrichedUnionRows,
  id: string | null | undefined,
): ModelStatsSelectedModel[] {
  const allModels = enrichedUnionRows.unionRows.map((unionRow) =>
    mapUnionModelToSelected(
      unionRow,
      enrichedUnionRows.openRouterSpeedById,
      enrichedUnionRows.openRouterPricingById,
      enrichedUnionRows.speedOutputTokenAnchors,
    ),
  );
  const modelsWithComputedPercentiles = withComputedPercentiles(allModels);
  const sortedModels = sortModelsByIntelligencePercentile(
    modelsWithComputedPercentiles,
  );
  const prunedModels = pruneSparseFields(sortedModels);
  return filterModelsById(prunedModels, id);
}
