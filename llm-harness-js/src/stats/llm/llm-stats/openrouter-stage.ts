/** OpenRouter enrichment stage for LLM stats: dedupe rows, backfill free costs, and fetch speed/pricing enrichments. */
import { getOpenRouterScrapedStats } from "../sources/openrouter-scraper";
import {
  PRIMARY_PROVIDER_ID,
  asFiniteNumber,
  asRecord,
  modelSlugFromModelId,
  normalizeProviderModelId,
  type JsonObject,
} from "../shared";

import { deriveSpeedOutputTokenAnchors } from "./scoring";
import {
  type EnrichedRows,
  type OpenRouterConfig,
  type ScoringConfig,
} from "./types";

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

function reasoningEffortPriority(
  aaSlug: string | null,
  canonicalSlug: string | null,
): number {
  if (aaSlug == null || canonicalSlug == null) {
    return 0;
  }
  const normalizedAaSlug = normalizeProviderModelId(aaSlug);
  const normalizedCanonicalSlug = normalizeProviderModelId(canonicalSlug);
  if (normalizedAaSlug === normalizedCanonicalSlug) {
    return 5;
  }
  const effortSuffixes = [
    ["-xhigh", 5],
    ["-high", 4],
    ["-medium", 3],
    ["-low", 2],
    ["-minimal", 1],
  ] as const;
  for (const [suffix, priority] of effortSuffixes) {
    if (normalizedAaSlug === `${normalizedCanonicalSlug}${suffix}`) {
      return priority;
    }
  }
  return 0;
}

function rowPriority(row: JsonObject, normalizedId: string): number {
  const providerId = row.provider_id;
  const openrouterBoost = providerId === PRIMARY_PROVIDER_ID ? 1_000_000 : 0;
  const intelligenceCostBoost = hasIntelligenceCost(row) ? 1_000 : 0;
  const scoreSignalBoost = hasScoreSignal(row) ? 10 : 0;
  const aaSlug = typeof row.aa_slug === "string" ? row.aa_slug : null;
  const canonicalSlug = modelSlugFromModelId(normalizedId);
  const reasoningEffortBoost =
    reasoningEffortPriority(aaSlug, canonicalSlug) * 10_000_000;
  return (
    reasoningEffortBoost +
    openrouterBoost +
    intelligenceCostBoost +
    scoreSignalBoost
  );
}

function dedupeRowsPreferOpenRouter(
  rows: Record<string, unknown>[],
): Record<string, unknown>[] {
  const groupedByNormalizedId = new Map<string, JsonObject[]>();
  const passthrough: Record<string, unknown>[] = [];

  for (const row of rows) {
    const rowRecord = asRecord(row);
    const id = typeof rowRecord.id === "string" ? rowRecord.id : null;
    if (!id) {
      passthrough.push(row);
      continue;
    }
    const key = normalizeProviderModelId(id);
    const group = groupedByNormalizedId.get(key) ?? [];
    group.push(rowRecord);
    groupedByNormalizedId.set(key, group);
  }

  const dedupedRows: JsonObject[] = [];
  for (const [normalizedId, group] of groupedByNormalizedId.entries()) {
    const winner = [...group].sort(
      (left, right) =>
        rowPriority(right, normalizedId) - rowPriority(left, normalizedId),
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

function backfillFreeModelCosts(
  rows: Record<string, unknown>[],
): Record<string, unknown>[] {
  const nonFreeCostById = new Map<string, JsonObject>();
  for (const row of rows) {
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

  return rows.map((row) => {
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

async function buildOpenRouterDataById(
  rows: Record<string, unknown>[],
  speedConcurrency: number,
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
      concurrency: speedConcurrency,
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

/** Fetch OpenRouter enrichments for the matched rows and return the late-bound speed/pricing maps. */
export async function enrichRows(
  matchedRows: Record<string, unknown>[],
  openrouterConfig: OpenRouterConfig,
  scoringConfig: ScoringConfig,
): Promise<EnrichedRows> {
  const dedupedRows = dedupeRowsPreferOpenRouter(matchedRows);
  const rows = backfillFreeModelCosts(dedupedRows);
  const { speedById: openRouterSpeedById, pricingById: openRouterPricingById } =
    await buildOpenRouterDataById(rows, openrouterConfig.speedConcurrency);
  const speedOutputTokenAnchors = deriveSpeedOutputTokenAnchors(
    openRouterSpeedById,
    scoringConfig,
  );
  return {
    rows,
    openRouterSpeedById,
    openRouterPricingById,
    speedOutputTokenAnchors,
  };
}
