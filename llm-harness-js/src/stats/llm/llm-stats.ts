/** Public LLM stats API: cache list payloads, rebuild from live sources when needed, and return failure-safe output. */
import {
  DEFAULT_OUTPUT_PATH,
  currentEpochSeconds,
  loadModelStatsSelectedFromCache,
  saveModelStatsSelectedToPath,
} from "./llm-stats/cache.js";
import { buildFinalModels } from "./llm-stats/final-stage.js";
import { enrichRows } from "./llm-stats/openrouter-stage.js";
import { buildMatchedRows } from "./llm-stats/match-stage.js";
import { fetchSourceData } from "./llm-stats/source-stage.js";
import {
  type LlmStatsStageConfig,
  type ModelStatsSelectedModel,
  type ModelStatsSelectedOptions,
  type ModelStatsSelectedPayload,
} from "./llm-stats/types.js";

export type {
  LlmStatsStageConfig,
  ModelStatsSelectedModel,
  ModelStatsSelectedOptions,
  ModelStatsSelectedPayload,
};

/** Centralized stage config for the LLM stats pipeline so matching, enrichment, pruning, and scoring tune from one place. */
export const LLM_STATS_STAGE_CONFIG = {
  matcher: {
    variantTokens: ["flash-lite", "flash", "pro", "nano", "mini", "lite"],
  },
  openrouter: {
    speedConcurrency: 8,
  },
  final: {
    nullFieldPruneThreshold: 0.5,
    nullFieldPruneRecentLookbackDays: 90,
  },
  scoring: {
    intelligenceBenchmarkKeys: [
      "omniscience_accuracy",
      "hle",
      "lcr",
      "scicode",
    ],
    agenticBenchmarkKeys: [
      "omniscience_nonhallucination_rate",
      "gdpval_normalized",
      "ifbench",
      "terminalbench_hard",
    ],
    defaultSpeedOutputTokenAnchors: [200, 500, 1_000, 2_000, 8_000],
    speedOutputTokenRangeMin: 200,
    speedOutputTokenRangeMax: 8_000,
    speedAnchorQuantiles: [0.25, 0.5, 0.75],
    weightedPriceInputRatio: 0.75,
    weightedPriceOutputRatio: 0.25,
  },
} satisfies LlmStatsStageConfig;

/** Persist the final model stats payload to disk while keeping write failures non-fatal. */
export async function saveModelStatsSelected(
  payload: ModelStatsSelectedPayload,
  outputPath = DEFAULT_OUTPUT_PATH,
): Promise<void> {
  await saveModelStatsSelectedToPath(payload, outputPath);
}

/** Build the final selected LLM stats payload with cache-first list mode and in-memory single-model mode. */
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

    const sourceData = await fetchSourceData();
    const matchedRows = await buildMatchedRows(
      sourceData,
      LLM_STATS_STAGE_CONFIG.matcher,
    );
    const enrichedRows = await enrichRows(
      matchedRows,
      LLM_STATS_STAGE_CONFIG.openrouter,
      LLM_STATS_STAGE_CONFIG.scoring,
    );
    const models = buildFinalModels(
      enrichedRows,
      options.id,
      LLM_STATS_STAGE_CONFIG.final,
      LLM_STATS_STAGE_CONFIG.scoring,
    );
    const fetchedAt = currentEpochSeconds();

    if (options.id != null) {
      return {
        fetched_at_epoch_seconds: fetchedAt,
        models,
      };
    }

    const listPayload: ModelStatsSelectedPayload = {
      fetched_at_epoch_seconds: fetchedAt,
      models,
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
