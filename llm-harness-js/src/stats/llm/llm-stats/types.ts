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

/** Matcher config controls how strict the scraper-to-models.dev ID matching stays around known model variant tokens. */
export type MatcherConfig = {
  /** Tokens that should match on both sides so `flash`, `pro`, `mini`, and similar variants do not collapse into the wrong canonical model. */
  variantTokens: readonly string[];
};

/** OpenRouter config controls the late enrichment fetches that fill speed and weighted pricing data after matching. */
export type OpenRouterConfig = {
  /** Maximum concurrent OpenRouter detail requests when scraping per-model speed and pricing data. */
  speedConcurrency: number;
};

/** Final-stage config controls how aggressively sparse fields are pruned from the public payload. */
export type FinalStageConfig = {
  /** Maximum null share allowed before a top-level or nested field is dropped from the final output sample. */
  nullFieldPruneThreshold: number;
  /** Recent-release window used to build the prune sample so newly added fields are judged against current models first. */
  nullFieldPruneRecentLookbackDays: number;
};

/** Scoring config collects the benchmark groups and heuristic weights used to derive raw and normalized score fields. */
export type ScoringConfig = {
  /** Benchmark keys averaged into the intelligence benchmark mean before it is blended with the intelligence index. */
  intelligenceBenchmarkKeys: readonly string[];
  /** Benchmark keys averaged into the agentic benchmark mean before it is blended with the agentic index. */
  agenticBenchmarkKeys: readonly string[];
  /** Fallback output-token anchors used when OpenRouter speed data is missing or too sparse to derive representative anchors. */
  defaultSpeedOutputTokenAnchors: readonly number[];
  /** Lower bound for remapping observed token anchors into a stable speed-scoring range. */
  speedOutputTokenRangeMin: number;
  /** Upper bound for remapping observed token anchors into a stable speed-scoring range. */
  speedOutputTokenRangeMax: number;
  /** Quantiles used alongside min and max to derive five representative speed anchors from OpenRouter observations. */
  speedAnchorQuantiles: readonly number[];
  /** Input-side weight for blended pricing so cached or weighted input costs can dominate the final price heuristic. */
  weightedPriceInputRatio: number;
  /** Output-side weight for blended pricing so output cost still contributes to the final price heuristic. */
  weightedPriceOutputRatio: number;
};

/** Full stage config bundles the main tuning knobs that shape matching, enrichment, pruning, and scoring behavior. */
export type LlmStatsStageConfig = {
  matcher: MatcherConfig;
  openrouter: OpenRouterConfig;
  final: FinalStageConfig;
  scoring: ScoringConfig;
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
