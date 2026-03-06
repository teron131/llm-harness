/** Shared public and intermediate types for the LLM matcher pipeline. */
import { getArtificialAnalysisStats } from "../sources/artificial-analysis-api";
import { getModelsDevStats } from "../sources/models-dev";

export type ModelsDevModel = Awaited<
  ReturnType<typeof getModelsDevStats>
>["models"][number];

export type ArtificialAnalysisModel = Awaited<
  ReturnType<typeof getArtificialAnalysisStats>
>["models"][number];

export type MatcherSourceModel = {
  sourceSlug: string;
  sourceName: string | null;
  sourceReleaseDate: string | null;
};

/** Candidate model from models.dev for one Artificial Analysis source model. */
export type LlmMatchCandidate = {
  model_id: string;
  provider_id: string;
  provider_name: string;
  model_name: string | null;
  score: number;
};

export type LlmMatchResult = LlmMatchCandidate | null;

/** Mapping entry for one Artificial Analysis model and its ranked match candidates. */
export type LlmMatchMappedModel = {
  artificial_analysis_slug: string;
  artificial_analysis_name: string | null;
  artificial_analysis_release_date: string | null;
  best_match: LlmMatchResult;
  candidates: LlmMatchCandidate[];
};

/** Full mapping payload for Artificial Analysis -> models.dev candidate resolution. */
export type LlmMatchModelMappingPayload = {
  artificial_analysis_fetched_at_epoch_seconds: number | null;
  models_dev_fetched_at_epoch_seconds: number | null;
  total_artificial_analysis_models: number;
  total_models_dev_models: number;
  max_candidates: number;
  void_mode: "maxmin_half";
  void_threshold: number | null;
  voided_count: number;
  models: LlmMatchMappedModel[];
};

/** Options for mapping generation and fallback-source injection. */
export type LlmMatchModelMappingOptions = {
  maxCandidates?: number;
  artificialAnalysisModels?: ArtificialAnalysisModel[];
  modelsDevModels?: ModelsDevModel[];
  scrapedRows?: unknown[];
};

export type LlmScraperFallbackMatchDiagnosticsPayload = {
  scraped_fetched_at_epoch_seconds: number | null;
  models_dev_fetched_at_epoch_seconds: number | null;
  total_scraped_models: number;
  total_models_dev_models: number;
  max_candidates: number;
  pre_void_matched_count: number;
  pre_void_unmatched_count: number;
  void_mode: "maxmin_half";
  void_threshold: number | null;
  voided_count: number;
  matched_count: number;
  unmatched_count: number;
  models: LlmMatchMappedModel[];
};

export type PreferredProviderPools = {
  primary: ModelsDevModel[];
  fallback: ModelsDevModel[];
};

export type MatcherRunOutput = {
  models: LlmMatchMappedModel[];
  voidThreshold: number | null;
  voidedCount: number;
  preVoidMatchedCount: number;
  preVoidUnmatchedCount: number;
  matchedCount: number;
  unmatchedCount: number;
};
