/** Public image stats API: cache list payloads, rebuild from live sources when needed, and return failure-safe output. */
import {
  currentEpochSeconds,
  DEFAULT_OUTPUT_PATH,
  loadImageStatsSelectedFromCache,
  saveImageStatsSelectedToPath,
} from "./image-stats/cache.js";
import { buildFinalModels } from "./image-stats/final-stage.js";
import { buildMatchedRows } from "./image-stats/match-stage.js";
import { fetchSourceData } from "./image-stats/source-stage.js";
import {
  type ImageStatsSelectedModel,
  type ImageStatsSelectedOptions,
  type ImageStatsSelectedPayload,
} from "./image-stats/types.js";

export type {
  ImageStatsSelectedModel,
  ImageStatsSelectedOptions,
  ImageStatsSelectedPayload,
};

export async function saveImageStatsSelected(
  payload: ImageStatsSelectedPayload,
  outputPath = DEFAULT_OUTPUT_PATH,
): Promise<void> {
  await saveImageStatsSelectedToPath(payload, outputPath);
}

export async function getImageStatsSelected(
  options: ImageStatsSelectedOptions = {},
): Promise<ImageStatsSelectedPayload> {
  try {
    if (options.id == null) {
      const cachedPayload =
        await loadImageStatsSelectedFromCache(DEFAULT_OUTPUT_PATH);
      if (cachedPayload) {
        return cachedPayload;
      }
    }

    const sourceData = await fetchSourceData();
    const matchedRows = await buildMatchedRows(sourceData);
    const models = buildFinalModels(matchedRows, options.id);
    const fetchedAt = currentEpochSeconds();

    if (options.id != null) {
      return {
        fetched_at_epoch_seconds: fetchedAt,
        models,
      };
    }

    const listPayload: ImageStatsSelectedPayload = {
      fetched_at_epoch_seconds: fetchedAt,
      models,
    };
    await saveImageStatsSelected(listPayload, DEFAULT_OUTPUT_PATH);
    return listPayload;
  } catch {
    return {
      fetched_at_epoch_seconds: null,
      models: [],
    };
  }
}
