/** Shared public and stage-handoff types for the final selected LLM stats pipeline. */
import { getArtificialAnalysisStats } from "../sources/artificial-analysis-api.js";
import { getModelsDevStats } from "../sources/models-dev.js";
import { type JsonObject } from "../shared.js";

export type ArtificialAnalysisModel = Awaited<
  ReturnType<typeof getArtificialAnalysisStats>
>["models"][number];

export type ModelsDevModel = Awaited<
  ReturnType<typeof getModelsDevStats>
>["models"][number];

export type ScrapedEvalModel = {
  model_id?: unknown;
  logo?: unknown;
  evaluations?: unknown;
  intelligence?: unknown;
  intelligence_index_cost?: unknown;
};

/** Final selected model row exposed by the stats API. */
export type ModelStatsSelectedModel = {
  id: string | null;
  name: string | null;
  provider: string | null;
  logo: string;
  attachment: boolean | null;
  reasoning: boolean | null;
  release_date: string | null;
  modalities: unknown;
  open_weights: boolean | null;
  cost: unknown;
  context_window: unknown;
  speed: JsonObject;
  intelligence: unknown;
  intelligence_index_cost: unknown;
  evaluations: unknown;
  scores: unknown;
  relative_scores: unknown;
};

/** Final model stats payload returned by the public stats API, with `null` fetch time when the pipeline fails safely. */
export type ModelStatsSelectedPayload = {
  fetched_at_epoch_seconds: number | null;
  models: ModelStatsSelectedModel[];
};

/** Options for model stats lookup: omit `id` for the full list or set it for exact-id filtering. */
export type ModelStatsSelectedOptions = {
  id?: string | null;
};

export type SourceData = {
  scrapedRows: unknown[];
  preferredModelsDevModels: ModelsDevModel[];
  modelsDevById: Map<string, ModelsDevModel>;
  apiBySlug: Map<string, ArtificialAnalysisModel>;
  scrapedBySlug: Map<string, ScrapedEvalModel>;
};

export type EnrichedRows = {
  rows: Record<string, unknown>[];
  openRouterSpeedById: Map<string, JsonObject>;
  openRouterPricingById: Map<string, JsonObject>;
  speedOutputTokenAnchors: number[];
};
