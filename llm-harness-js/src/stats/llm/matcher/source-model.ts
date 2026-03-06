import { asRecord, modelSlugFromModelId } from "../shared.js";

import {
  type ArtificialAnalysisModel,
  type MatcherSourceModel,
} from "./types.js";

export function buildSourceModelsFromArtificialAnalysis(
  artificialAnalysisModels: ArtificialAnalysisModel[],
): MatcherSourceModel[] {
  return artificialAnalysisModels.map((artificialAnalysisModel) => ({
    sourceSlug:
      typeof artificialAnalysisModel.slug === "string"
        ? artificialAnalysisModel.slug
        : "",
    sourceName:
      typeof artificialAnalysisModel.name === "string"
        ? artificialAnalysisModel.name
        : null,
    sourceReleaseDate:
      typeof artificialAnalysisModel.release_date === "string"
        ? artificialAnalysisModel.release_date
        : null,
  }));
}

export function buildSourceModelsFromScrapedRows(
  scrapedRows: unknown[],
): MatcherSourceModel[] {
  return scrapedRows.map((scrapedRow) => {
    const scrapedRowRecord = asRecord(scrapedRow);
    const modelId =
      typeof scrapedRowRecord.model_id === "string"
        ? scrapedRowRecord.model_id
        : null;
    const sourceSlug = modelSlugFromModelId(modelId) ?? "";
    return {
      sourceSlug,
      sourceName: modelId,
      sourceReleaseDate: null,
    };
  });
}
