import { getArtificialAnalysisStats } from "./sources/artificial-analysis-api.js";
import { getArtificialAnalysisScrapedEvalsOnlyStats } from "./sources/artificial-analysis-scraper.js";
import { getModelsDevStats } from "./sources/models-dev.js";

import {
  buildInputModelsFromArtificialAnalysis,
  buildInputModelsFromScrapedRows,
  runMatcher,
  scopeToPreferredProviderModels,
  uniqueModelCount,
} from "./matcher/pipeline.js";
import {
  type LlmMatchModelMappingOptions,
  type LlmMatchModelMappingPayload,
  type LlmScraperFallbackMatchDiagnosticsPayload,
} from "./matcher/types.js";

export type {
  LlmMatchCandidate,
  LlmMatchMappedModel,
  LlmMatchModelMappingOptions,
  LlmMatchModelMappingPayload,
  LlmMatchResult,
  LlmScraperFallbackMatchDiagnosticsPayload,
} from "./matcher/types.js";

const DEFAULT_MAX_CANDIDATES = 5;

/**
 * Build candidate mappings from Artificial Analysis models to models.dev models.
 */
export async function getMatchModelMapping(
  options: LlmMatchModelMappingOptions = {},
): Promise<LlmMatchModelMappingPayload> {
  const maxCandidates = options.maxCandidates ?? DEFAULT_MAX_CANDIDATES;
  const artificialAnalysisStats =
    options.artificialAnalysisModels != null
      ? {
          fetched_at_epoch_seconds: null,
          models: options.artificialAnalysisModels,
        }
      : await getArtificialAnalysisStats();
  const modelsDevStats =
    options.modelsDevModels != null
      ? {
          fetched_at_epoch_seconds: null,
          models: options.modelsDevModels,
        }
      : await getModelsDevStats();

  const scopedModels = scopeToPreferredProviderModels(modelsDevStats.models);
  const totalScopedModels = uniqueModelCount([
    ...scopedModels.primary,
    ...scopedModels.fallback,
  ]);
  const inputModels = buildInputModelsFromArtificialAnalysis(
    artificialAnalysisStats.models,
  );
  const matcherOutput = runMatcher(inputModels, scopedModels, maxCandidates);

  return {
    artificial_analysis_fetched_at_epoch_seconds:
      artificialAnalysisStats.fetched_at_epoch_seconds,
    models_dev_fetched_at_epoch_seconds:
      modelsDevStats.fetched_at_epoch_seconds,
    total_artificial_analysis_models: matcherOutput.models.length,
    total_models_dev_models: totalScopedModels,
    max_candidates: maxCandidates,
    void_mode: "maxmin_half",
    void_threshold: matcherOutput.voidThreshold,
    voided_count: matcherOutput.voidedCount,
    models: matcherOutput.models,
  };
}

/**
 * Run the same matcher algorithm using scraper-only AA models (API-keyless fallback).
 */
export async function getScraperFallbackMatchDiagnostics(
  options: LlmMatchModelMappingOptions = {},
): Promise<LlmScraperFallbackMatchDiagnosticsPayload> {
  const maxCandidates = options.maxCandidates ?? DEFAULT_MAX_CANDIDATES;
  const scrapedStats =
    options.scrapedRows != null
      ? {
          fetched_at_epoch_seconds: null,
          data: options.scrapedRows,
        }
      : await getArtificialAnalysisScrapedEvalsOnlyStats();
  const modelsDevStats =
    options.modelsDevModels != null
      ? {
          fetched_at_epoch_seconds: null,
          models: options.modelsDevModels,
        }
      : await getModelsDevStats();

  const scopedModels = scopeToPreferredProviderModels(modelsDevStats.models);
  const totalScopedModels = uniqueModelCount([
    ...scopedModels.primary,
    ...scopedModels.fallback,
  ]);
  const inputModels = buildInputModelsFromScrapedRows(scrapedStats.data);
  const matcherOutput = runMatcher(inputModels, scopedModels, maxCandidates);

  return {
    scraped_fetched_at_epoch_seconds: scrapedStats.fetched_at_epoch_seconds,
    models_dev_fetched_at_epoch_seconds:
      modelsDevStats.fetched_at_epoch_seconds,
    total_scraped_models: scrapedStats.data.length,
    total_models_dev_models: totalScopedModels,
    max_candidates: maxCandidates,
    pre_void_matched_count: matcherOutput.preVoidMatchedCount,
    pre_void_unmatched_count: matcherOutput.preVoidUnmatchedCount,
    void_mode: "maxmin_half",
    void_threshold: matcherOutput.voidThreshold,
    voided_count: matcherOutput.voidedCount,
    matched_count: matcherOutput.matchedCount,
    unmatched_count: matcherOutput.unmatchedCount,
    models: matcherOutput.models,
  };
}
