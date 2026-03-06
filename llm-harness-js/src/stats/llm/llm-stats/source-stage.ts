import { getArtificialAnalysisStats } from "../sources/artificial-analysis-api.js";
import { getArtificialAnalysisScrapedEvalsOnlyStats } from "../sources/artificial-analysis-scraper.js";
import { getModelsDevStats } from "../sources/models-dev.js";
import {
  FALLBACK_PROVIDER_IDS,
  PRIMARY_PROVIDER_ID,
  modelSlugFromModelId,
} from "../shared.js";

import {
  type ArtificialAnalysisModel,
  type ModelsDevModel,
  type ScrapedEvalModel,
  type SourceData,
} from "./types.js";

function dedupePreferredProviderModels(
  modelsDevModels: ModelsDevModel[],
): ModelsDevModel[] {
  const preferredModels = modelsDevModels.filter(
    (modelsDevModel) =>
      modelsDevModel.provider_id === PRIMARY_PROVIDER_ID ||
      FALLBACK_PROVIDER_IDS.has(modelsDevModel.provider_id),
  );
  const byModelId = new Map<string, ModelsDevModel>();
  const withPriority = preferredModels.map((modelsDevModel) => ({
    modelsDevModel,
    priority: modelsDevModel.provider_id === PRIMARY_PROVIDER_ID ? 0 : 1,
  }));
  withPriority.sort((left, right) => left.priority - right.priority);
  for (const { modelsDevModel } of withPriority) {
    byModelId.set(
      modelsDevModel.model_id,
      byModelId.get(modelsDevModel.model_id) ?? modelsDevModel,
    );
  }
  return [...byModelId.values()];
}

function buildModelsDevById(
  modelsDevModels: ModelsDevModel[],
): Map<string, ModelsDevModel> {
  return new Map(
    modelsDevModels.map((modelsDevModel) => [
      modelsDevModel.model_id,
      modelsDevModel,
    ]),
  );
}

function buildScrapedBySlug(
  scrapedRows: unknown[],
): Map<string, ScrapedEvalModel> {
  const scrapedBySlug = new Map<string, ScrapedEvalModel>();
  for (const scrapedRow of scrapedRows) {
    const scrapedModel = scrapedRow as ScrapedEvalModel;
    const artificialAnalysisSlug = modelSlugFromModelId(scrapedModel.model_id);
    if (artificialAnalysisSlug) {
      scrapedBySlug.set(artificialAnalysisSlug, scrapedModel);
    }
  }
  return scrapedBySlug;
}

function buildApiBySlug(
  artificialAnalysisModels: ArtificialAnalysisModel[],
): Map<string, ArtificialAnalysisModel> {
  return new Map(
    artificialAnalysisModels
      .map(
        (model) =>
          [
            typeof model.slug === "string" && model.slug.length > 0
              ? model.slug
              : null,
            model,
          ] as const,
      )
      .filter(
        (entry): entry is [string, ArtificialAnalysisModel] => entry[0] != null,
      ),
  );
}

export async function fetchSourceData(): Promise<SourceData> {
  const [
    artificialAnalysisApiStats,
    artificialAnalysisScrapedStats,
    modelsDevStats,
  ] = await Promise.all([
    getArtificialAnalysisStats(),
    getArtificialAnalysisScrapedEvalsOnlyStats(),
    getModelsDevStats(),
  ]);
  const preferredModelsDevModels = dedupePreferredProviderModels(
    modelsDevStats.models,
  );
  return {
    scrapedRows: artificialAnalysisScrapedStats.data,
    preferredModelsDevModels,
    modelsDevById: buildModelsDevById(preferredModelsDevModels),
    apiBySlug: buildApiBySlug(artificialAnalysisApiStats.models),
    scrapedBySlug: buildScrapedBySlug(artificialAnalysisScrapedStats.data),
  };
}
