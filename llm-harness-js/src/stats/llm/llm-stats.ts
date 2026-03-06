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
  type ModelStatsSelectedModel,
  type ModelStatsSelectedOptions,
  type ModelStatsSelectedPayload,
} from "./llm-stats/types.js";

export type {
  ModelStatsSelectedModel,
  ModelStatsSelectedOptions,
  ModelStatsSelectedPayload,
};

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
    const matchedRows = await buildMatchedRows(sourceData);
    const enrichedRows = await enrichRows(matchedRows);
    const models = buildFinalModels(enrichedRows, options.id);
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
