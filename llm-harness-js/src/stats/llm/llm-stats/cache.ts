import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { type ModelStatsSelectedPayload } from "./types.js";

export const DEFAULT_OUTPUT_PATH = resolve(".cache/model_stats.json");
const CACHE_DIR = resolve(".cache");
const CACHE_TTL_SECONDS = 60 * 60 * 24;

async function writeJson(path: string, payload: unknown): Promise<void> {
  await mkdir(CACHE_DIR, { recursive: true });
  await writeFile(path, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
}

function nowEpochSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

function isFreshCache(fetchedAtEpochSeconds: unknown): boolean {
  if (typeof fetchedAtEpochSeconds !== "number") {
    return false;
  }
  const ageSeconds = nowEpochSeconds() - fetchedAtEpochSeconds;
  return ageSeconds >= 0 && ageSeconds <= CACHE_TTL_SECONDS;
}

export function currentEpochSeconds(): number {
  return nowEpochSeconds();
}

export async function saveModelStatsSelectedToPath(
  payload: ModelStatsSelectedPayload,
  outputPath = DEFAULT_OUTPUT_PATH,
): Promise<void> {
  try {
    await writeJson(outputPath, payload);
  } catch {
    // Intentionally swallow cache write errors: API remains in-memory first.
  }
}

export async function loadModelStatsSelectedFromCache(
  outputPath: string,
): Promise<ModelStatsSelectedPayload | null> {
  try {
    const content = await readFile(outputPath, "utf-8");
    const payload = JSON.parse(content) as ModelStatsSelectedPayload;
    if (!Array.isArray(payload.models)) {
      return null;
    }
    if (!isFreshCache(payload.fetched_at_epoch_seconds)) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}
