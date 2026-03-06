/** Match stage for LLM stats: turn scraper-first matcher diagnostics into merged source rows. */
import { getScraperFallbackMatchDiagnostics } from "../matcher.js";
import {
  asRecord,
  modelSlugFromModelId,
  normalizeProviderModelId,
  type JsonObject,
} from "../shared.js";

import {
  type ArtificialAnalysisModel,
  type MatcherConfig,
  type ModelsDevModel,
  type ScrapedEvalModel,
  type SourceData,
} from "./types.js";

function hasToken(id: string, token: string) {
  return id.includes(token);
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

function hasVariantConflict(
  artificialAnalysisSlug: string,
  matchedModelId: string,
  matcherConfig: MatcherConfig,
): boolean {
  const aa = normalizeProviderModelId(artificialAnalysisSlug);
  const matched = normalizeProviderModelId(matchedModelId);
  return matcherConfig.variantTokens.some(
    (token) => hasToken(aa, token) !== hasToken(matched, token),
  );
}

/** Build one matched row from scraper data, then fill missing nested fields from the AA API row when available. */
function buildMatchedRowFromScrapedModel(
  scrapedModel: ScrapedEvalModel,
  apiModel: ArtificialAnalysisModel | null,
  matchedModelId: string,
  modelsDevById: Map<string, ModelsDevModel>,
): Record<string, unknown> {
  const artificialAnalysisModelId =
    typeof scrapedModel.model_id === "string" ? scrapedModel.model_id : null;
  const artificialAnalysisSlug = modelSlugFromModelId(
    artificialAnalysisModelId,
  );
  const evaluations = asRecord(scrapedModel.evaluations);
  const intelligence = asRecord(scrapedModel.intelligence);
  const intelligenceIndexCost = asRecord(scrapedModel.intelligence_index_cost);
  const logo = typeof scrapedModel.logo === "string" ? scrapedModel.logo : null;
  const apiEvaluations = asRecord(apiModel?.evaluations);
  const apiIntelligence = asRecord(apiModel?.intelligence);
  const apiIntelligenceIndexCost = asRecord(apiModel?.intelligence_index_cost);
  const matchedModelsDev = modelsDevById.get(matchedModelId) ?? null;
  const matchedModelFields = asRecord(matchedModelsDev?.model);
  const canonicalId = canonicalModelId(
    matchedModelsDev?.model?.id ?? matchedModelId,
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
        : artificialAnalysisModelId,
    aa_id: artificialAnalysisModelId,
    aa_slug: artificialAnalysisSlug,
    family: matchedFamily,
    logo,
    ...matchedModelFieldsWithoutIdFamilyAndModelRefs,
    evaluations: {
      ...apiEvaluations,
      ...evaluations,
    },
    intelligence: {
      ...apiIntelligence,
      ...intelligence,
    },
    intelligence_index_cost: {
      ...apiIntelligenceIndexCost,
      ...intelligenceIndexCost,
    },
  };
}

/** Build matched intermediate rows by running scraper fallback diagnostics and rejecting obvious variant mismatches. */
export async function buildMatchedRows(
  sourceData: SourceData,
  matcherConfig: MatcherConfig,
): Promise<Record<string, unknown>[]> {
  const fallbackDiagnostics = await getScraperFallbackMatchDiagnostics({
    scrapedRows: sourceData.scrapedRows,
    modelsDevModels: sourceData.preferredModelsDevModels,
  });

  return fallbackDiagnostics.models
    .map((matchedModel) => {
      const matchedModelId = matchedModel.best_match?.model_id;
      if (typeof matchedModelId !== "string" || matchedModelId.length === 0) {
        return null;
      }
      if (
        hasVariantConflict(
          matchedModel.artificial_analysis_slug,
          matchedModelId,
          matcherConfig,
        )
      ) {
        return null;
      }
      const scrapedModel = sourceData.scrapedBySlug.get(
        matchedModel.artificial_analysis_slug,
      );
      if (!scrapedModel) {
        return null;
      }
      const apiModel =
        sourceData.apiBySlug.get(matchedModel.artificial_analysis_slug) ?? null;
      return buildMatchedRowFromScrapedModel(
        scrapedModel,
        apiModel,
        matchedModelId,
        sourceData.modelsDevById,
      );
    })
    .filter((row): row is Record<string, unknown> => row != null);
}
