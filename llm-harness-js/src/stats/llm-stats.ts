import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { getArtificialAnalysisStats } from "./data-sources/llm/artificial-analysis-api";
import { getArtificialAnalysisScrapedEvalsOnlyStats } from "./data-sources/llm/artificial-analysis-scraper";
import {
  getMatchModelMapping,
  getScraperFallbackMatchDiagnostics,
} from "./data-sources/llm/matcher";
import { getModelsDevStats } from "./data-sources/llm/models-dev";

const DEFAULT_OUTPUT_PATH = resolve(".cache/model_stats.json");
const CACHE_DIR = resolve(".cache");
const CACHE_TTL_SECONDS = 60 * 60 * 24;

type JsonObject = Record<string, unknown>;
type ModelsDevModel = Awaited<
  ReturnType<typeof getModelsDevStats>
>["models"][number];
type ArtificialAnalysisModel = Awaited<
  ReturnType<typeof getArtificialAnalysisStats>
>["models"][number];
type ScrapedEvalModel = {
  model_id?: unknown;
  logo?: unknown;
  evaluations?: unknown;
  intelligence?: unknown;
};

const PRIMARY_PROVIDER_FILTER = "openrouter" as const;
const FALLBACK_PROVIDER_FILTERS = new Set(["openai", "google", "anthropic"]);

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
  apiKey?: string;
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

function canonicalModelId(
  modelId: unknown,
  providerId: unknown,
  fallbackModelId: unknown,
): string | null {
  if (typeof modelId === "string" && modelId.includes("/")) {
    return modelId;
  }
  if (typeof providerId === "string" && typeof modelId === "string") {
    return `${providerId}/${modelId}`;
  }
  if (typeof providerId === "string" && typeof fallbackModelId === "string") {
    return `${providerId}/${fallbackModelId}`;
  }
  return typeof modelId === "string" ? modelId : null;
}

function providerFromModel(model: JsonObject): string | null {
  const fromId = providerFromId(model.id);
  if (fromId) {
    return fromId;
  }
  return typeof model.provider_id === "string" ? model.provider_id : null;
}

function asRecord(value: unknown): JsonObject {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
}

function asFiniteNumber(value: unknown): number | null {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : null;
}

function buildEvaluations(model: JsonObject): unknown {
  const evaluations = asRecord(model.evaluations);
  return Object.keys(evaluations).length > 0 ? evaluations : null;
}

function buildScores(model: JsonObject): unknown {
  const baseScores = asRecord(model.scores);
  const intelligence = asRecord(model.intelligence);
  const evaluations = asRecord(model.evaluations);
  const agenticScore =
    asFiniteNumber(intelligence.agentic_index) ??
    asFiniteNumber(evaluations.agentic_index) ??
    asFiniteNumber(evaluations.artificial_analysis_agentic_index);
  if (agenticScore == null) {
    return model.scores ?? null;
  }
  return {
    ...baseScores,
    agentic_score: agenticScore,
  };
}

function normalize(value: string): string {
  return value
    .toLowerCase()
    .replace(/[._:\s]+/g, "-")
    .replace(/[^a-z0-9/-]+/g, "")
    .replace(/-+/g, "-")
    .replace(/^[-/]+|[-/]+$/g, "");
}

function creatorSlugFromApiModel(
  model: ArtificialAnalysisModel,
): string | null {
  const creator = asRecord(model.model_creator);
  if (typeof creator.slug === "string" && creator.slug.length > 0) {
    return creator.slug;
  }
  if (typeof creator.name === "string" && creator.name.length > 0) {
    return normalize(creator.name);
  }
  return null;
}

function artificialAnalysisModelId(
  model: ArtificialAnalysisModel,
): string | null {
  const creatorSlug = creatorSlugFromApiModel(model);
  const modelSlug = typeof model.slug === "string" ? model.slug : null;
  if (!creatorSlug || !modelSlug) {
    return null;
  }
  return `${creatorSlug}/${modelSlug}`;
}

function modelSlugFromModelId(modelId: unknown): string | null {
  if (typeof modelId !== "string" || modelId.length === 0) {
    return null;
  }
  const slug = modelId.split("/").at(-1);
  return slug && slug.length > 0 ? slug : null;
}

function scopeToPreferredProviderModels(
  modelsDevModels: ModelsDevModel[],
): ModelsDevModel[] {
  const preferredModels = modelsDevModels.filter(
    (modelStatsModel) =>
      modelStatsModel.provider_id === PRIMARY_PROVIDER_FILTER ||
      FALLBACK_PROVIDER_FILTERS.has(modelStatsModel.provider_id),
  );
  const byModelId = new Map<string, ModelsDevModel>();
  const withPriority = preferredModels.map((model) => ({
    model,
    priority: model.provider_id === PRIMARY_PROVIDER_FILTER ? 0 : 1,
  }));
  withPriority.sort((left, right) => left.priority - right.priority);
  for (const { model } of withPriority) {
    byModelId.set(model.model_id, byModelId.get(model.model_id) ?? model);
  }
  return [...byModelId.values()];
}

function buildModelsDevById(
  modelsDevModels: ModelsDevModel[],
): Map<string, ModelsDevModel> {
  return new Map(modelsDevModels.map((model) => [model.model_id, model]));
}

function buildScrapedByModelId(
  scrapedModels: unknown[],
): Map<string, ScrapedEvalModel> {
  const scrapedByModelId = new Map<string, ScrapedEvalModel>();
  for (const model of scrapedModels) {
    const typedModel = model as ScrapedEvalModel;
    if (
      typeof typedModel.model_id === "string" &&
      typedModel.model_id.length > 0
    ) {
      scrapedByModelId.set(typedModel.model_id, typedModel);
    }
  }
  return scrapedByModelId;
}

function buildUnionFromApiMatch(
  evalModel: ArtificialAnalysisModel,
  bestMatchModelId: string,
  modelsDevById: Map<string, ModelsDevModel>,
  scrapedByModelId: Map<string, ScrapedEvalModel>,
): Record<string, unknown> {
  const modelId = artificialAnalysisModelId(evalModel);
  const scrapedModel = modelId ? scrapedByModelId.get(modelId) : undefined;
  const apiEvaluations = asRecord(evalModel.evaluations);
  const scrapedEvaluations = asRecord(scrapedModel?.evaluations);
  const mergedEvaluations = {
    ...apiEvaluations,
    ...scrapedEvaluations,
  };
  const intelligence = asRecord(scrapedModel?.intelligence);
  const logo =
    typeof scrapedModel?.logo === "string" ? scrapedModel.logo : null;
  const matchedModelsDev = modelsDevById.get(bestMatchModelId) ?? null;
  const matchedModelFields = asRecord(matchedModelsDev?.model);
  const canonicalId = canonicalModelId(
    matchedModelsDev?.model?.id ?? bestMatchModelId,
    matchedModelsDev?.provider_id,
    matchedModelsDev?.model_id,
  );
  const {
    id: _matchedId,
    name: _matchedName,
    family: matchedFamily,
    model_id: _matchedModelId,
    slug: _matchedSlug,
    ...matchedModelFieldsWithoutIdFamilyAndModelRefs
  } = matchedModelFields;
  const evalModelFields = asRecord(evalModel);
  const {
    name: _evalName,
    slug: _evalSlug,
    evaluations: _evalEvaluations,
    model_creator: _evalModelCreator,
    ...evalModelFieldsWithoutIdentity
  } = evalModelFields;

  return {
    id: canonicalId,
    provider_id: matchedModelsDev?.provider_id ?? null,
    ...matchedModelFieldsWithoutIdFamilyAndModelRefs,
    ...evalModelFieldsWithoutIdentity,
    openrouter_id: matchedModelsDev?.model?.id ?? null,
    name:
      typeof matchedModelsDev?.model?.name === "string"
        ? matchedModelsDev.model.name
        : typeof evalModel.name === "string"
          ? evalModel.name
          : null,
    aa_id: modelId,
    aa_slug: typeof evalModel.slug === "string" ? evalModel.slug : null,
    family: matchedFamily,
    logo,
    evaluations: mergedEvaluations,
    intelligence,
  };
}

function buildUnionFromScrapedMatch(
  scrapedRow: ScrapedEvalModel,
  bestMatchModelId: string,
  modelsDevById: Map<string, ModelsDevModel>,
): Record<string, unknown> {
  const modelId =
    typeof scrapedRow.model_id === "string" ? scrapedRow.model_id : null;
  const slug = modelSlugFromModelId(modelId);
  const evaluations = asRecord(scrapedRow.evaluations);
  const intelligence = asRecord(scrapedRow.intelligence);
  const logo = typeof scrapedRow.logo === "string" ? scrapedRow.logo : null;
  const matchedModelsDev = modelsDevById.get(bestMatchModelId) ?? null;
  const matchedModelFields = asRecord(matchedModelsDev?.model);
  const canonicalId = canonicalModelId(
    matchedModelsDev?.model?.id ?? bestMatchModelId,
    matchedModelsDev?.provider_id,
    matchedModelsDev?.model_id,
  );
  const {
    id: _matchedId,
    name: _matchedName,
    family: matchedFamily,
    model_id: _matchedModelId,
    slug: _matchedSlug,
    ...matchedModelFieldsWithoutIdFamilyAndModelRefs
  } = matchedModelFields;

  return {
    id: canonicalId,
    provider_id: matchedModelsDev?.provider_id ?? null,
    openrouter_id: matchedModelsDev?.model?.id ?? null,
    name:
      typeof matchedModelsDev?.model?.name === "string"
        ? matchedModelsDev.model.name
        : modelId,
    aa_id: modelId,
    aa_slug: slug,
    family: matchedFamily,
    logo,
    ...matchedModelFieldsWithoutIdFamilyAndModelRefs,
    evaluations,
    intelligence,
  };
}

async function buildUnionModelsFromPrimaryPathWithApiKey(
  apiKey?: string,
): Promise<Record<string, unknown>[]> {
  const [
    artificialAnalysisStats,
    artificialAnalysisScrapedStats,
    modelsDevStats,
  ] = await Promise.all([
    getArtificialAnalysisStats(apiKey ? { apiKey } : {}),
    getArtificialAnalysisScrapedEvalsOnlyStats(),
    getModelsDevStats(),
  ]);

  if (artificialAnalysisStats.models.length === 0) {
    return [];
  }

  const scopedModelsDevModels = scopeToPreferredProviderModels(
    modelsDevStats.models,
  );
  const matchMapping = await getMatchModelMapping({
    artificialAnalysisModels: artificialAnalysisStats.models,
    modelsDevModels: scopedModelsDevModels,
  });
  const modelsDevById = buildModelsDevById(scopedModelsDevModels);
  const scrapedByModelId = buildScrapedByModelId(
    artificialAnalysisScrapedStats.data,
  );
  const matchBySlug = new Map(
    matchMapping.models.map((model) => [model.artificial_analysis_slug, model]),
  );

  return artificialAnalysisStats.models
    .map((evalModel) => {
      const slug = typeof evalModel.slug === "string" ? evalModel.slug : "";
      const match = matchBySlug.get(slug);
      const bestMatchModelId = match?.best_match?.model_id;
      if (
        typeof bestMatchModelId !== "string" ||
        bestMatchModelId.length === 0
      ) {
        return null;
      }
      return buildUnionFromApiMatch(
        evalModel,
        bestMatchModelId,
        modelsDevById,
        scrapedByModelId,
      );
    })
    .filter((row): row is Record<string, unknown> => row != null);
}

async function buildUnionModelsFromFallbackPath(): Promise<
  Record<string, unknown>[]
> {
  const [scrapedStats, modelsDevStats] = await Promise.all([
    getArtificialAnalysisScrapedEvalsOnlyStats(),
    getModelsDevStats(),
  ]);
  const scopedModelsDevModels = scopeToPreferredProviderModels(
    modelsDevStats.models,
  );
  const fallbackDiagnostics = await getScraperFallbackMatchDiagnostics({
    scrapedRows: scrapedStats.data,
    modelsDevModels: scopedModelsDevModels,
  });
  const modelsDevById = buildModelsDevById(scopedModelsDevModels);
  const scrapedBySlug = new Map<string, ScrapedEvalModel>();
  for (const row of scrapedStats.data) {
    const typedRow = row as ScrapedEvalModel;
    const slug = modelSlugFromModelId(typedRow.model_id);
    if (slug) {
      scrapedBySlug.set(slug, typedRow);
    }
  }

  return fallbackDiagnostics.models
    .map((model) => {
      const bestMatchModelId = model.best_match?.model_id;
      if (
        typeof bestMatchModelId !== "string" ||
        bestMatchModelId.length === 0
      ) {
        return null;
      }
      const scrapedRow = scrapedBySlug.get(model.artificial_analysis_slug);
      if (!scrapedRow) {
        return null;
      }
      return buildUnionFromScrapedMatch(
        scrapedRow,
        bestMatchModelId,
        modelsDevById,
      );
    })
    .filter((row): row is Record<string, unknown> => row != null);
}

async function buildUnionModelsWithApiKey(
  apiKey?: string,
): Promise<Record<string, unknown>[]> {
  const primaryModels = await buildUnionModelsFromPrimaryPathWithApiKey(apiKey);
  if (primaryModels.length > 0) {
    return primaryModels;
  }
  return buildUnionModelsFromFallbackPath();
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
  const provider = providerFromModel(model);
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
    evaluations: buildEvaluations(model),
    scores: buildScores(model),
    percentiles: model.percentiles ?? null,
  };
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

    const unionModels = await buildUnionModelsWithApiKey(options.apiKey);
    const allModels = unionModels.map(mapUnionModelToSelected);
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
