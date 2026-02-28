import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const MODELS_DEV_URL = "https://models.dev/api.json";
const CACHE_PATH = resolve(".cache/models_dev_api.json");
const OUTPUT_PATH = resolve(".cache/models_dev_output.json");
const LOOKBACK_DAYS = 365;
const REQUEST_TIMEOUT_MS = 30_000;
const DEFAULT_CACHE_TTL_SECONDS = 60 * 60 * 12;
const CACHE_DIR = resolve(".cache");

type NumberOrNull = number | null;

export type ModelRecord = {
  id?: string;
  name?: string;
  family?: string;
  release_date?: string;
  last_updated?: string;
  open_weights?: boolean;
  reasoning?: boolean;
  tool_call?: boolean;
  cost?: {
    input?: number;
    output?: number;
    cache_read?: number;
    cache_write?: number;
    output_audio?: number;
  };
  limit?: {
    context?: number;
    output?: number;
  };
  modalities?: {
    input?: string[];
    output?: string[];
  };
  [key: string]: unknown;
};

export type ProviderRecord = {
  id?: string;
  name?: string;
  api?: string;
  models?: Record<string, ModelRecord>;
  [key: string]: unknown;
};

export type ModelsDevPayload = Record<string, ProviderRecord>;

export type ModelsDevFlatModel = {
  provider_id: string;
  provider_name: string;
  model_id: string;
  model: ModelRecord;
};

type ModelsDevCachePayload = {
  fetched_at_epoch_seconds: number;
  status_code: number;
  payload: ModelsDevPayload;
};

export type ModelsDevOutputPayload = {
  fetched_at_epoch_seconds: number;
  status_code: number;
  models: ModelsDevFlatModel[];
};

export type ModelStatsOptions = {
  refreshCache?: boolean;
  cacheTtlSeconds?: number;
};

function nowEpochSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

function isoDateDaysAgo(days: number): string {
  return new Date(Date.now() - days * 24 * 60 * 60 * 1000)
    .toISOString()
    .slice(0, 10);
}

function isRecentDate(
  isoDate: string | undefined,
  cutoffIsoDate: string,
): boolean {
  if (!isoDate) {
    return false;
  }
  return isoDate >= cutoffIsoDate;
}

function asFinite(value: unknown): NumberOrNull {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

async function writeJson(path: string, payload: unknown): Promise<void> {
  await mkdir(CACHE_DIR, { recursive: true });
  await writeFile(path, JSON.stringify(payload, null, 2), "utf-8");
}

async function loadCache(): Promise<ModelsDevCachePayload> {
  const content = await readFile(CACHE_PATH, "utf-8");
  return JSON.parse(content) as ModelsDevCachePayload;
}

async function fetchAndCacheModelsDev(): Promise<ModelsDevCachePayload> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const response = await fetch(MODELS_DEV_URL, { signal: controller.signal });
  clearTimeout(timeout);

  if (!response.ok) {
    throw new Error(`models.dev request failed: ${response.status}`);
  }

  const payload = (await response.json()) as ModelsDevPayload;
  const cachePayload: ModelsDevCachePayload = {
    fetched_at_epoch_seconds: nowEpochSeconds(),
    status_code: response.status,
    payload,
  };
  await writeJson(CACHE_PATH, cachePayload);
  return cachePayload;
}

async function resolveCachePayload(
  refreshCache: boolean,
  cacheTtlSeconds: number,
): Promise<ModelsDevCachePayload> {
  if (refreshCache) {
    return fetchAndCacheModelsDev();
  }

  try {
    const cached = await loadCache();
    const ageSeconds = nowEpochSeconds() - cached.fetched_at_epoch_seconds;
    if (ageSeconds <= cacheTtlSeconds) {
      return cached;
    }
    return fetchAndCacheModelsDev();
  } catch {
    return fetchAndCacheModelsDev();
  }
}

function flattenModels(payload: ModelsDevPayload): ModelsDevFlatModel[] {
  const rows: ModelsDevFlatModel[] = [];
  for (const [providerId, provider] of Object.entries(payload)) {
    const providerName = provider.name ?? providerId;
    const models = provider.models ?? {};
    for (const [modelId, model] of Object.entries(models)) {
      rows.push({
        provider_id: providerId,
        provider_name: providerName,
        model_id: model.id ?? modelId,
        model,
      });
    }
  }
  return rows;
}

function rankRecentModels(
  models: ModelsDevFlatModel[],
  cutoffIsoDate: string,
): ModelsDevFlatModel[] {
  return models
    .filter((row) => isRecentDate(row.model.release_date, cutoffIsoDate))
    .sort((left, right) => {
      const leftOutputCost =
        asFinite(left.model.cost?.output) ?? Number.POSITIVE_INFINITY;
      const rightOutputCost =
        asFinite(right.model.cost?.output) ?? Number.POSITIVE_INFINITY;
      if (leftOutputCost !== rightOutputCost) {
        return leftOutputCost - rightOutputCost;
      }
      return (right.model.release_date ?? "").localeCompare(
        left.model.release_date ?? "",
      );
    });
}

export async function getModelStats(
  options: ModelStatsOptions = {},
): Promise<ModelsDevOutputPayload> {
  const refreshCache = options.refreshCache ?? false;
  const cacheTtlSeconds = options.cacheTtlSeconds ?? DEFAULT_CACHE_TTL_SECONDS;
  const cachePayload = await resolveCachePayload(refreshCache, cacheTtlSeconds);
  const cutoffIsoDate = isoDateDaysAgo(LOOKBACK_DAYS);
  const outputPayload: ModelsDevOutputPayload = {
    fetched_at_epoch_seconds: cachePayload.fetched_at_epoch_seconds,
    status_code: cachePayload.status_code,
    models: rankRecentModels(
      flattenModels(cachePayload.payload),
      cutoffIsoDate,
    ),
  };

  await writeJson(OUTPUT_PATH, outputPayload);
  return outputPayload;
}
