import { getOpenRouterScrapedStats } from "../sources/openrouter-scraper.js";
import { asFiniteNumber, asRecord, type JsonObject } from "../shared.js";

import {
  backfillFreeModelCosts,
  dedupeRowsPreferOpenRouter,
} from "./cleanup.js";
import { deriveSpeedOutputTokenAnchors } from "./scoring.js";
import { type EnrichedUnionRows } from "./types.js";

const OPENROUTER_SPEED_CONCURRENCY = 8;

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
