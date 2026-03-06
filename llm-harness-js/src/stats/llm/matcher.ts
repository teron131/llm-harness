/** Public matcher APIs for LLM model mapping and scraper fallback diagnostics. */
import { getArtificialAnalysisStats } from "./sources/artificial-analysis-api.js";
import { getArtificialAnalysisScrapedEvalsOnlyStats } from "./sources/artificial-analysis-scraper.js";
import { getModelsDevStats } from "./sources/models-dev.js";

import {
  runMatcher,
  splitPreferredProviderModels,
  uniqueModelCount,
} from "./matcher/pipeline.js";
import {
  buildSourceModelsFromArtificialAnalysis,
  buildSourceModelsFromScrapedRows,
} from "./matcher/source-model.js";
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

/** Build candidate mappings from Artificial Analysis API rows to models.dev identities. */
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

  const providerPools = splitPreferredProviderModels(modelsDevStats.models);
  const totalScopedModels = uniqueModelCount([
    ...providerPools.primary,
    ...providerPools.fallback,
  ]);
  const sourceModels = buildSourceModelsFromArtificialAnalysis(
    artificialAnalysisStats.models,
  );
  const matcherOutput = runMatcher(sourceModels, providerPools, maxCandidates);

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

/** Run the same matcher algorithm against scraper rows for the API-keyless fallback path. */
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

  const providerPools = splitPreferredProviderModels(modelsDevStats.models);
  const totalScopedModels = uniqueModelCount([
    ...providerPools.primary,
    ...providerPools.fallback,
  ]);
  const sourceModels = buildSourceModelsFromScrapedRows(scrapedStats.data);
  const matcherOutput = runMatcher(sourceModels, providerPools, maxCandidates);

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
