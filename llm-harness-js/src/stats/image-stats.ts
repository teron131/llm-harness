import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { getImageModelsUnion } from "./data-sources/image/matcher";

const DEFAULT_OUTPUT_PATH = resolve(".cache/image_stats.json");
const CACHE_DIR = resolve(".cache");
const CACHE_TTL_SECONDS = 60 * 60 * 24;

type JsonObject = Record<string, unknown>;
type NumberOrNull = number | null;

export type ImageStatsSelectedModel = {
  id: string | null;
  name: string | null;
  provider: string | null;
  logo: string;
  release_date: string | null;
  sources: {
    artificial_analysis: boolean;
    arena_ai: boolean;
  };
  source_scores: {
    artificial_analysis: JsonObject | null;
    arena_ai: JsonObject | null;
  };
  source_percentiles: {
    artificial_analysis: JsonObject | null;
    arena_ai: JsonObject | null;
  };
  scores: {
    photorealistic_score: NumberOrNull;
    illustrative_score: NumberOrNull;
    contextual_score: NumberOrNull;
    overall_score: NumberOrNull;
  };
  percentiles: {
    photorealistic_percentile: NumberOrNull;
    illustrative_percentile: NumberOrNull;
    contextual_percentile: NumberOrNull;
    overall_percentile: NumberOrNull;
  };
};

export type ImageStatsSelectedPayload = {
  fetched_at_epoch_seconds: number | null;
  models: ImageStatsSelectedModel[];
};

export type ImageStatsSelectedOptions = {
  id?: string | null;
};

function asRecord(value: unknown): JsonObject {
  return value != null && typeof value === "object"
    ? (value as JsonObject)
    : {};
}

function meanOrNull(values: unknown[]): NumberOrNull {
  const finite = values
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
  if (finite.length === 0) {
    return null;
  }
  return Number(
    (finite.reduce((sum, value) => sum + value, 0) / finite.length).toFixed(4),
  );
}

function toModelId(value: string): string {
  return value
    .toLowerCase()
    .replace(/[._:/\s]+/g, "-")
    .replace(/[^a-z0-9-]+/g, "")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function providerFromArenaProvider(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const left = value.split("·")[0]?.trim();
  return left && left.length > 0 ? left : null;
}

function buildLogo(model: JsonObject, provider: string | null): string {
  const artificialAnalysis = asRecord(model.artificial_analysis);
  const modelCreator = asRecord(artificialAnalysis.model_creator);
  const logoSlug = modelCreator.slug;
  if (typeof logoSlug === "string" && logoSlug.length > 0) {
    return `https://artificialanalysis.ai/img/logos/${logoSlug}_small.svg`;
  }
  return `https://models.dev/logos/${(provider ?? "unknown").toLowerCase()}.svg`;
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

async function writeJson(path: string, payload: unknown): Promise<void> {
  await mkdir(CACHE_DIR, { recursive: true });
  await writeFile(path, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
}

async function loadImageStatsSelectedFromCache(
  outputPath: string,
): Promise<ImageStatsSelectedPayload | null> {
  try {
    const content = await readFile(outputPath, "utf-8");
    const payload = JSON.parse(content) as ImageStatsSelectedPayload;
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

function pickAaPercentiles(model: JsonObject): JsonObject | null {
  const artificialAnalysis = asRecord(model.artificial_analysis);
  const percentiles = asRecord(artificialAnalysis.percentiles);
  return Object.keys(percentiles).length > 0 ? percentiles : null;
}

function pickArenaPercentiles(model: JsonObject): JsonObject | null {
  const arena = asRecord(model.arena_ai);
  const percentiles = asRecord(arena.percentiles);
  return Object.keys(percentiles).length > 0 ? percentiles : null;
}

function pickAaScores(model: JsonObject): JsonObject | null {
  const artificialAnalysis = asRecord(model.artificial_analysis);
  const weightedScores = asRecord(artificialAnalysis.weighted_scores);
  return Object.keys(weightedScores).length > 0 ? weightedScores : null;
}

function pickArenaScores(model: JsonObject): JsonObject | null {
  const arena = asRecord(model.arena_ai);
  const weightedScores = asRecord(arena.weighted_scores);
  return Object.keys(weightedScores).length > 0 ? weightedScores : null;
}

function mapUnionModelToSelected(unionModel: unknown): ImageStatsSelectedModel {
  const model = asRecord(unionModel);
  const artificialAnalysis = asRecord(model.artificial_analysis);
  const arena = asRecord(model.arena_ai);
  const artificialAnalysisScores = pickAaScores(model);
  const arenaScores = pickArenaScores(model);
  const artificialAnalysisPercentiles = pickAaPercentiles(model);
  const arenaPercentiles = pickArenaPercentiles(model);
  const bestMatch = asRecord(model.best_match);
  const inferredId = toModelId(
    (typeof artificialAnalysis.slug === "string" && artificialAnalysis.slug) ||
      (typeof artificialAnalysis.name === "string" &&
        artificialAnalysis.name) ||
      (typeof arena.model === "string" && arena.model) ||
      (typeof bestMatch.arena_model === "string" && bestMatch.arena_model) ||
      "unknown",
  );
  const provider =
    (typeof artificialAnalysis.model_creator === "object" &&
    artificialAnalysis.model_creator != null &&
    typeof (artificialAnalysis.model_creator as JsonObject).name === "string"
      ? ((artificialAnalysis.model_creator as JsonObject).name as string)
      : null) ?? providerFromArenaProvider(arena.provider);

  const photorealisticScore = meanOrNull([
    artificialAnalysisScores?.photorealistic,
    arenaScores?.photorealistic,
  ]);
  const illustrativeScore = meanOrNull([
    artificialAnalysisScores?.illustrative,
    arenaScores?.illustrative,
  ]);
  const contextualScore = meanOrNull([
    artificialAnalysisScores?.contextual,
    arenaScores?.contextual,
  ]);
  const overallScore = meanOrNull([
    photorealisticScore,
    illustrativeScore,
    contextualScore,
  ]);
  const photorealisticPercentile = meanOrNull([
    artificialAnalysisPercentiles?.photorealistic_percentile,
    arenaPercentiles?.photorealistic_percentile,
  ]);
  const illustrativePercentile = meanOrNull([
    artificialAnalysisPercentiles?.illustrative_percentile,
    arenaPercentiles?.illustrative_percentile,
  ]);
  const contextualPercentile = meanOrNull([
    artificialAnalysisPercentiles?.contextual_percentile,
    arenaPercentiles?.contextual_percentile,
  ]);
  const overallPercentile = meanOrNull([
    photorealisticPercentile,
    illustrativePercentile,
    contextualPercentile,
  ]);

  return {
    id: inferredId.length > 0 ? inferredId : null,
    name:
      (typeof artificialAnalysis.name === "string" &&
        artificialAnalysis.name) ||
      (typeof artificialAnalysis.slug === "string" &&
        artificialAnalysis.slug) ||
      (typeof arena.model === "string" && arena.model) ||
      (typeof bestMatch.arena_model === "string" && bestMatch.arena_model) ||
      null,
    provider: provider ?? null,
    logo: buildLogo(model, provider),
    release_date:
      typeof artificialAnalysis.release_date === "string"
        ? artificialAnalysis.release_date
        : null,
    sources: {
      artificial_analysis: Object.keys(artificialAnalysis).length > 0,
      arena_ai: Object.keys(arena).length > 0,
    },
    source_scores: {
      artificial_analysis: artificialAnalysisScores,
      arena_ai: arenaScores,
    },
    source_percentiles: {
      artificial_analysis: artificialAnalysisPercentiles,
      arena_ai: arenaPercentiles,
    },
    scores: {
      photorealistic_score: photorealisticScore,
      illustrative_score: illustrativeScore,
      contextual_score: contextualScore,
      overall_score: overallScore,
    },
    percentiles: {
      photorealistic_percentile: photorealisticPercentile,
      illustrative_percentile: illustrativePercentile,
      contextual_percentile: contextualPercentile,
      overall_percentile: overallPercentile,
    },
  };
}

function filterModelsById(
  models: ImageStatsSelectedModel[],
  id: string | null | undefined,
): ImageStatsSelectedModel[] {
  if (id == null) {
    return models;
  }
  return models.filter((model) => model.id === id);
}

export async function saveImageStatsSelected(
  payload: ImageStatsSelectedPayload,
  outputPath = DEFAULT_OUTPUT_PATH,
): Promise<void> {
  try {
    await writeJson(outputPath, payload);
  } catch {
    // Intentionally swallow cache write errors: API remains in-memory first.
  }
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

    const matchUnion = await getImageModelsUnion();
    const allModels = matchUnion.models
      .map(mapUnionModelToSelected)
      .sort(
        (left, right) =>
          (right.scores.overall_score ?? Number.NEGATIVE_INFINITY) -
          (left.scores.overall_score ?? Number.NEGATIVE_INFINITY),
      );
    const filteredModels = filterModelsById(allModels, options.id);
    const fetchedAt = nowEpochSeconds();

    if (options.id != null) {
      return {
        fetched_at_epoch_seconds: fetchedAt,
        models: filteredModels,
      };
    }

    const listPayload: ImageStatsSelectedPayload = {
      fetched_at_epoch_seconds: fetchedAt,
      models: filteredModels,
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
