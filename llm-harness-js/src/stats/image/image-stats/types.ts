import type {
  ArtificialAnalysisImageEnrichedModel,
  ArtificialAnalysisImageOutputPayload,
} from "../sources/artificial-analysis.js";
import type {
  ArenaAiImageAggregatedModel,
  ArenaAiImageOutputPayload,
} from "../sources/arena-ai.js";
import type { ImageMatchCandidate } from "../matcher.js";
import type { JsonObject, NumberOrNull } from "../../utils.js";

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

export type ImageSourceData = {
  artificialAnalysisPayload: ArtificialAnalysisImageOutputPayload;
  arenaPayload: ArenaAiImageOutputPayload;
  artificialAnalysisModelsBySlug: Map<
    string,
    ArtificialAnalysisImageEnrichedModel
  >;
  arenaModelsByName: Map<string, ArenaAiImageAggregatedModel>;
};

export type ImageUnionRow = {
  artificial_analysis_slug: string | null;
  artificial_analysis_name: string | null;
  artificial_analysis_provider: string | null;
  best_match: ImageMatchCandidate | null;
  candidates: ImageMatchCandidate[];
  artificial_analysis: ArtificialAnalysisImageEnrichedModel | null;
  arena_ai: ArenaAiImageAggregatedModel | null;
};
