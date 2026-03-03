import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { getMatchModelsUnion } from "./data-sources/matcher";

const DEFAULT_OUTPUT_PATH = resolve(".cache/model_stats.json");
const CACHE_DIR = resolve(".cache");
const CACHE_TTL_SECONDS = 60 * 60 * 24;

type JsonObject = Record<string, unknown>;

/**
 * Final selected model row exposed by the stats API.
 */
export type ModelStatsSelectedModel = {
  id: string | null;
  name: string | null;
  provider: string | null;
  logo: string;
  attachment: boolean | null;
  reasoning: boolean | null;
  release_date: string | null;
  modalities: unknown;
  open_weights: boolean | null;
  cost: unknown;
  context_window: unknown;
  speed: JsonObject;
  evaluations: unknown;
  scores: unknown;
  percentiles: unknown;
};

/**
 * Final model stats payload returned by the public stats API.
 *
 * `fetched_at_epoch_seconds` is `null` when the upstream pipeline fails and
 * the function returns an empty-safe payload.
 */
export type ModelStatsSelectedPayload = {
  fetched_at_epoch_seconds: number | null;
  models: ModelStatsSelectedModel[];
};

/**
 * Options for model stats lookup.
 *
 * - omit `id` (or set `null`) to return the full list
 * - set `id` to return exact-id matches only
 */
export type ModelStatsSelectedOptions = {
  id?: string | null;
};

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

function asRecord(value: unknown): JsonObject {
  return value != null && typeof value === "object"
    ? (value as JsonObject)
    : {};
}

function buildLogo(model: JsonObject, provider: string | null): string {
  const modelCreator = asRecord(model.model_creator);
  const logoSlug = modelCreator.slug;
  if (typeof logoSlug === "string" && logoSlug.length > 0) {
    return `https://artificialanalysis.ai/img/logos/${logoSlug}_small.svg`;
  }
  return `https://models.dev/logos/${provider ?? "unknown"}.svg`;
}

function buildSpeed(model: JsonObject): JsonObject {
  return Object.fromEntries(
    Object.entries(model).filter(([key]) => key.startsWith("median_")),
  );
}

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

function filterModelsById(
  models: ModelStatsSelectedModel[],
  id: string | null | undefined,
): ModelStatsSelectedModel[] {
  if (id == null) {
    return models;
  }
  return models.filter((model) => model.id === id);
}

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
  try {
    await writeJson(outputPath, payload);
  } catch {
    // Intentionally swallow cache write errors: API remains in-memory first.
  }
}

async function loadModelStatsSelectedFromCache(
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

function mapUnionModelToSelected(unionModel: unknown): ModelStatsSelectedModel {
  const model = asRecord(unionModel);
  const provider = providerFromId(model.id);
  return {
    id: typeof model.id === "string" ? model.id : null,
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
    cost: model.cost ?? null,
    context_window: model.limit ?? null,
    speed: buildSpeed(model),
    evaluations: model.evaluations ?? null,
    scores: model.scores ?? null,
    percentiles: model.percentiles ?? null,
  };
}

/**
 * Return final model stats enriched from matcher union data.
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

    const matchUnion = await getMatchModelsUnion();
    const allModels = matchUnion.models.map(mapUnionModelToSelected);
    const filteredModels = filterModelsById(allModels, options.id);
    const fetchedAt = nowEpochSeconds();

    if (options.id != null) {
      return {
        fetched_at_epoch_seconds: fetchedAt,
        models: filteredModels,
      };
    }

    const listPayload: ModelStatsSelectedPayload = {
      fetched_at_epoch_seconds: fetchedAt,
      models: filteredModels,
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
