/** Cache helpers for the final selected LLM stats payload. */
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import {
  isFreshEpochSeconds,
  nowEpochSeconds,
  writeJsonFile,
} from "../../utils";
import type { ModelStatsSelectedPayload } from "./types";

export const DEFAULT_OUTPUT_PATH = resolve(".cache/llm_stats.json");
const CACHE_TTL_SECONDS = 60 * 60 * 24;

export function currentEpochSeconds(): number {
  return nowEpochSeconds();
}

/** Save the final payload to disk and swallow write failures to keep the API failure-safe. */
export async function saveModelStatsSelectedToPath(
  payload: ModelStatsSelectedPayload,
  outputPath = DEFAULT_OUTPUT_PATH,
): Promise<void> {
  try {
    await writeJsonFile(outputPath, payload);
  } catch {
    // Intentionally swallow cache write errors: API remains in-memory first.
  }
}

/** Load a fresh cached list payload from disk or return `null` when it is missing, invalid, or stale. */
export async function loadModelStatsSelectedFromCache(
  outputPath: string,
): Promise<ModelStatsSelectedPayload | null> {
  try {
    const content = await readFile(outputPath, "utf-8");
    const payload = JSON.parse(content) as ModelStatsSelectedPayload;
    if (!Array.isArray(payload.models)) {
      return null;
    }
    if (
      !isFreshEpochSeconds(payload.fetched_at_epoch_seconds, CACHE_TTL_SECONDS)
    ) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}
