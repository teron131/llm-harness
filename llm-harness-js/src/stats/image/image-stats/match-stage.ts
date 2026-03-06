import { getImageMatchModelMapping } from "../matcher.js";
import { asRecord } from "../../utils.js";

import type { ImageSourceData, ImageUnionRow } from "./types.js";

function mergeImageRow(
  sourceData: ImageSourceData,
  mappedModel: Pick<
    ImageUnionRow,
    | "artificial_analysis_slug"
    | "artificial_analysis_name"
    | "artificial_analysis_provider"
    | "best_match"
    | "candidates"
  >,
): ImageUnionRow {
  const arenaModelName =
    typeof mappedModel.best_match?.arena_model === "string"
      ? mappedModel.best_match.arena_model
      : null;

  return {
    ...mappedModel,
    artificial_analysis:
      mappedModel.artificial_analysis_slug != null
        ? (sourceData.artificialAnalysisModelsBySlug.get(
            mappedModel.artificial_analysis_slug,
          ) ?? null)
        : null,
    arena_ai:
      arenaModelName != null
        ? (sourceData.arenaModelsByName.get(arenaModelName) ?? null)
        : null,
  };
}

export async function buildMatchedRows(
  sourceData: ImageSourceData,
): Promise<ImageUnionRow[]> {
  const mapping = await getImageMatchModelMapping({
    artificialAnalysisModels: sourceData.artificialAnalysisPayload.data,
    arenaModels: sourceData.arenaPayload.rows,
  });
  const matchedRows = mapping.models.map((model) =>
    mergeImageRow(sourceData, model),
  );
  const matchedArenaNames = new Set(
    mapping.models
      .map((model) => asRecord(model.best_match).arena_model)
      .filter((value): value is string => typeof value === "string"),
  );
  const unmatchedArenaRows = sourceData.arenaPayload.rows
    .filter((model) => !matchedArenaNames.has(model.model))
    .map(
      (model): ImageUnionRow => ({
        artificial_analysis_slug: null,
        artificial_analysis_name: null,
        artificial_analysis_provider: null,
        best_match: null,
        candidates: [],
        artificial_analysis: null,
        arena_ai: model,
      }),
    );

  return [...matchedRows, ...unmatchedArenaRows];
}
