import {
  DEFAULT_OUTPUT_PATH,
  currentEpochSeconds,
  loadModelStatsSelectedFromCache,
  saveModelStatsSelectedToPath,
} from "./llm-stats/cache.js";
import {
  buildFinalModels,
  buildMatchedRows,
  enrichRows,
  fetchSourceData,
} from "./llm-stats/pipeline.js";
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
  await saveModelStatsSelectedToPath(payload, outputPath);
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
