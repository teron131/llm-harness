import { asRecord, meanOrNull, type JsonObject } from "../../utils.js";

import type { ImageStatsSelectedModel, ImageUnionRow } from "./types.js";

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

function pickAaPercentiles(model: JsonObject): JsonObject | null {
  const percentiles = asRecord(asRecord(model.artificial_analysis).percentiles);
  return Object.keys(percentiles).length > 0 ? percentiles : null;
}

function pickArenaPercentiles(model: JsonObject): JsonObject | null {
  const percentiles = asRecord(asRecord(model.arena_ai).percentiles);
  return Object.keys(percentiles).length > 0 ? percentiles : null;
}

function pickAaScores(model: JsonObject): JsonObject | null {
  const weightedScores = asRecord(
    asRecord(model.artificial_analysis).weighted_scores,
  );
  return Object.keys(weightedScores).length > 0 ? weightedScores : null;
}

function pickArenaScores(model: JsonObject): JsonObject | null {
  const weightedScores = asRecord(asRecord(model.arena_ai).weighted_scores);
  return Object.keys(weightedScores).length > 0 ? weightedScores : null;
}

function mapUnionModelToSelected(
  unionModel: ImageUnionRow,
): ImageStatsSelectedModel {
  const model = unionModel as unknown as JsonObject;
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
    typeof asRecord(artificialAnalysis.model_creator).name === "string"
      ? (asRecord(artificialAnalysis.model_creator).name as string)
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

export function buildFinalModels(
  unionModels: ImageUnionRow[],
  id?: string | null,
): ImageStatsSelectedModel[] {
  const selectedModels = unionModels
    .map(mapUnionModelToSelected)
    .sort(
      (left, right) =>
        (right.scores.overall_score ?? Number.NEGATIVE_INFINITY) -
        (left.scores.overall_score ?? Number.NEGATIVE_INFINITY),
    );
  return filterModelsById(selectedModels, id);
}
