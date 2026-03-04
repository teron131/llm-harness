import { getArtificialAnalysisStats } from "./artificial-analysis-api";
import { getArtificialAnalysisScrapedEvalsOnlyStats } from "./artificial-analysis-scraper";
import { getModelsDevStats } from "./models-dev";

type ModelsDevModel = Awaited<
  ReturnType<typeof getModelsDevStats>
>["models"][number];
type ArtificialAnalysisModel = Awaited<
  ReturnType<typeof getArtificialAnalysisStats>
>["models"][number];

const TOKEN_PREFIX_WEIGHTS = [5, 4, 3, 2, 1] as const;
const DEFAULT_MAX_CANDIDATES = 5;
const TOKEN_PREFIX_REWARD_MULTIPLIER = 2;
const NUMERIC_EXACT_MATCH_REWARD = 2;
const NUMERIC_CLOSENESS_REWARD_SCALE = 0.1;
const NUMERIC_ALL_EQUAL_REWARD = 0.2;
const VARIANT_SUFFIX_REWARD = 2;
const COVERAGE_EXACT_REWARD = 4;
const COVERAGE_MISSING_BASE_PENALTY = 1;
const B_SCALE_EXACT_REWARD = 3;
const B_SCALE_MISMATCH_PENALTY = 4;
const B_SCALE_MISSING_PENALTY = 2;
const ACTIVE_B_EXACT_REWARD = 2;
const ACTIVE_B_MISMATCH_PENALTY = 2;
const CHAR_PREFIX_REWARD_SCALE = 0.03;
const LENGTH_GAP_PENALTY_SCALE = 0.005;
const PRIMARY_PROVIDER_FILTER = "openrouter" as const;
const FALLBACK_PROVIDER_FILTERS = new Set(["openai", "google", "anthropic"]);
const VOID_THRESHOLD_RANGE_RATIO = 0.35;
// Model-name noise tags that frequently appear as variants/capabilities rather
// than core model identity, so we exclude them from matching tokens.
const MODEL_NAME_TAG_TOKENS = new Set([
  "free",
  "extended",
  "exacto",
  "instruct",
  "vl",
  "thinking",
  "reasoning",
  "online",
  "nitro",
]);

/**
 * Candidate model from models.dev for a given Artificial Analysis model.
 */
export type LlmMatchCandidate = {
  model_id: string;
  provider_id: string;
  provider_name: string;
  model_name: string | null;
  score: number;
};

export type LlmMatchResult = LlmMatchCandidate | null;

/**
 * Mapping entry for one Artificial Analysis model and its match candidates.
 */
export type LlmMatchMappedModel = {
  artificial_analysis_slug: string;
  artificial_analysis_name: string | null;
  artificial_analysis_release_date: string | null;
  best_match: LlmMatchResult;
  candidates: LlmMatchCandidate[];
};

/**
 * Full mapping payload (Artificial Analysis -> models.dev candidates).
 */
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

/**
 * Options for mapping generation.
 */
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

function normalize(value: string): string {
  return value
    .toLowerCase()
    .replace(/[._:\s]+/g, "-")
    .replace(/[^a-z0-9/-]+/g, "")
    .replace(/-+/g, "-")
    .replace(/^[-/]+|[-/]+$/g, "");
}

function asRecord(value: unknown): Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function modelSlugFromModelId(modelId: unknown): string | null {
  if (typeof modelId !== "string" || modelId.length === 0) {
    return null;
  }
  const slug = modelId.split("/").at(-1);
  return slug && slug.length > 0 ? slug : null;
}

function splitBaseModelId(modelId: string): string {
  const parts = modelId.split("/");
  return parts.at(-1) ?? modelId;
}

function isBScaleToken(token: string): boolean {
  return /^\d+b$/.test(token) || /^a\d+b$/.test(token);
}

function splitMixedAlphaNumericToken(token: string): string[] {
  if (isBScaleToken(token)) {
    return [token];
  }
  return token.split(/(?<=\D)(?=\d)|(?<=\d)(?=\D)/g).filter(Boolean);
}

function splitTokens(value: string): string[] {
  return normalize(value)
    .split("-")
    .flatMap((token) => splitMixedAlphaNumericToken(token))
    .filter((token) => token && !MODEL_NAME_TAG_TOKENS.has(token));
}

function firstParsedNumber(
  tokens: string[],
  parser: (token: string | undefined) => number | null,
): number | null {
  for (const token of tokens) {
    const parsedValue = parser(token);
    if (parsedValue != null) {
      return parsedValue;
    }
  }
  return null;
}

function isNumericToken(token: string | undefined): boolean {
  return Boolean(token && /^\d+$/.test(token));
}

function parseNumericOrBScaleToken(token: string | undefined): number | null {
  if (!token) {
    return null;
  }
  if (/^\d+$/.test(token)) {
    return Number(token);
  }
  const billionMatch = /^(\d+)b$/.exec(token);
  if (billionMatch) {
    return Number(billionMatch[1]);
  }
  const aBillionMatch = /^a(\d+)b$/.exec(token);
  if (aBillionMatch) {
    return Number(aBillionMatch[1]);
  }
  return null;
}

function parseBScaleToken(token: string | undefined): number | null {
  if (!token) {
    return null;
  }
  const billionMatch = /^(\d+)b$/.exec(token);
  return billionMatch ? Number(billionMatch[1]) : null;
}

function parseActiveBToken(token: string | undefined): number | null {
  if (!token) {
    return null;
  }
  const activeMatch = /^a(\d+)b$/.exec(token);
  return activeMatch ? Number(activeMatch[1]) : null;
}

function commonPrefixLength(left: string, right: string): number {
  const maxLength = Math.min(left.length, right.length);
  let index = 0;
  while (index < maxLength && left[index] === right[index]) {
    index += 1;
  }
  return index;
}

function weightedTokenPrefixScore(
  leftTokens: string[],
  rightTokens: string[],
): number {
  const maxLength = Math.min(leftTokens.length, rightTokens.length);
  let score = 0;
  for (let index = 0; index < maxLength; index += 1) {
    if (leftTokens[index] !== rightTokens[index]) {
      break;
    }
    score += TOKEN_PREFIX_WEIGHTS[index] ?? 0;
  }
  return score;
}

function numericMatchReward(evalSlug: string, modelId: string): number {
  const evalTokens = splitTokens(evalSlug);
  const modelTokens = splitTokens(splitBaseModelId(modelId));
  const maxLength = Math.min(evalTokens.length, modelTokens.length);
  for (let index = 0; index < maxLength; index += 1) {
    const evalToken = evalTokens[index];
    const modelToken = modelTokens[index];
    const evalNumericValue = parseNumericOrBScaleToken(evalToken);
    const modelNumericValue = parseNumericOrBScaleToken(modelToken);
    if (evalNumericValue != null && modelNumericValue != null) {
      return evalNumericValue === modelNumericValue
        ? NUMERIC_EXACT_MATCH_REWARD
        : 0;
    }
  }
  return 0;
}

function numericClosenessReward(evalSlug: string, modelId: string): number {
  const evalNumbers = splitTokens(evalSlug)
    .map((token) => parseNumericOrBScaleToken(token))
    .filter((value): value is number => value != null);
  const modelNumbers = splitTokens(splitBaseModelId(modelId))
    .map((token) => parseNumericOrBScaleToken(token))
    .filter((value): value is number => value != null);

  const maxLength = Math.max(evalNumbers.length, modelNumbers.length);
  for (let index = 0; index < maxLength; index += 1) {
    const evalValue = evalNumbers[index];
    const modelValue = modelNumbers[index];
    if (evalValue == null || modelValue == null) {
      return 0;
    }
    if (evalValue === modelValue) {
      continue;
    }
    return (
      NUMERIC_CLOSENESS_REWARD_SCALE / (1 + Math.abs(evalValue - modelValue))
    );
  }
  return NUMERIC_ALL_EQUAL_REWARD;
}

function bScaleRewardOrPenalty(
  evalSlug: string,
  modelId: string,
  modelName: string,
): number {
  const evalTokens = splitTokens(evalSlug);
  const modelBaseTokens = splitTokens(splitBaseModelId(modelId));
  const modelNameTokens = splitTokens(modelName);

  const evalBScale = firstParsedNumber(evalTokens, parseBScaleToken);
  if (evalBScale == null) {
    return 0;
  }

  const baseBScale = firstParsedNumber(modelBaseTokens, parseBScaleToken);
  const nameBScale = firstParsedNumber(modelNameTokens, parseBScaleToken);
  const candidateBScale = baseBScale ?? nameBScale;
  if (candidateBScale == null) {
    return -B_SCALE_MISSING_PENALTY;
  }
  if (candidateBScale === evalBScale) {
    return B_SCALE_EXACT_REWARD;
  }
  return -B_SCALE_MISMATCH_PENALTY;
}

function hasHardBScaleMismatch(
  evalSlug: string,
  modelId: string,
  modelName: string,
): boolean {
  const evalBScale = firstParsedNumber(splitTokens(evalSlug), parseBScaleToken);
  if (evalBScale == null) {
    return false;
  }

  const modelBaseBScale = firstParsedNumber(
    splitTokens(splitBaseModelId(modelId)),
    parseBScaleToken,
  );
  const modelNameBScale = firstParsedNumber(
    splitTokens(modelName),
    parseBScaleToken,
  );
  const candidateBScale = modelBaseBScale ?? modelNameBScale;
  if (candidateBScale == null) {
    return false;
  }

  return candidateBScale !== evalBScale;
}

function activeBRewardOrPenalty(
  evalSlug: string,
  modelId: string,
  modelName: string,
): number {
  const evalTokens = splitTokens(evalSlug);
  const modelBaseTokens = splitTokens(splitBaseModelId(modelId));
  const modelNameTokens = splitTokens(modelName);

  const evalActiveB = firstParsedNumber(evalTokens, parseActiveBToken);
  if (evalActiveB == null) {
    return 0;
  }

  const baseActiveB = firstParsedNumber(modelBaseTokens, parseActiveBToken);
  const nameActiveB = firstParsedNumber(modelNameTokens, parseActiveBToken);
  const candidateActiveB = baseActiveB ?? nameActiveB;
  if (candidateActiveB == null) {
    return 0;
  }
  if (candidateActiveB === evalActiveB) {
    return ACTIVE_B_EXACT_REWARD;
  }
  return -ACTIVE_B_MISMATCH_PENALTY;
}

function sameVariantReward(
  evalSlug: string,
  modelId: string,
  modelName: string,
): number {
  const evalTokens = splitTokens(evalSlug);
  const modelBaseTokens = splitTokens(splitBaseModelId(modelId));
  const modelNameTokens = splitTokens(modelName);
  const evalLastToken = evalTokens.at(-1);
  if (!evalLastToken || isNumericToken(evalLastToken)) {
    return 0;
  }
  const baseLastToken = modelBaseTokens.at(-1);
  const nameLastToken = modelNameTokens.at(-1);
  if (evalLastToken === baseLastToken || evalLastToken === nameLastToken) {
    return VARIANT_SUFFIX_REWARD;
  }
  return 0;
}

function coverageRewardOrPenalty(
  evalSlug: string,
  modelId: string,
  modelName: string,
): number {
  const evalSet = new Set(splitTokens(evalSlug));
  const baseSet = new Set(splitTokens(splitBaseModelId(modelId)));
  const nameSet = new Set(splitTokens(modelName));

  function compareSets(candidateSet: Set<string>): number {
    if (evalSet.size === 0) {
      return 0;
    }
    const missingCount = [...evalSet].filter(
      (token) => !candidateSet.has(token),
    ).length;
    if (missingCount > 0) {
      return -COVERAGE_MISSING_BASE_PENALTY - missingCount;
    }
    if (candidateSet.size === evalSet.size) {
      return COVERAGE_EXACT_REWARD;
    }
    return 0;
  }

  return Math.max(compareSets(baseSet), compareSets(nameSet));
}

function hasFirstTokenMatch(
  evalSlug: string,
  modelId: string,
  modelName: string,
): boolean {
  // Guardrail: first-token mismatch usually means wrong model family.
  const evalFirst = splitTokens(evalSlug)[0];
  if (!evalFirst) {
    return false;
  }
  return (
    evalFirst === splitTokens(splitBaseModelId(modelId))[0] ||
    evalFirst === splitTokens(modelName)[0]
  );
}

function scoreCandidate(
  evalSlug: string,
  modelId: string,
  modelName: string,
): number {
  // Prefix reward addresses cross-family false positives.
  const normalizedEval = normalize(evalSlug);
  const normalizedModelBase = normalize(splitBaseModelId(modelId));
  const normalizedModelName = normalize(modelName);
  const evalTokens = splitTokens(evalSlug);
  const modelBaseTokens = splitTokens(splitBaseModelId(modelId));
  const modelNameTokens = splitTokens(modelName);
  const prefixBase = commonPrefixLength(normalizedEval, normalizedModelBase);
  const prefixName = commonPrefixLength(normalizedEval, normalizedModelName);
  const maxPrefix = Math.max(prefixBase, prefixName);
  if (maxPrefix === 0) {
    return 0;
  }
  if (hasHardBScaleMismatch(evalSlug, modelId, modelName)) {
    return 0;
  }

  const weightedTokenScore = Math.max(
    weightedTokenPrefixScore(evalTokens, modelBaseTokens),
    weightedTokenPrefixScore(evalTokens, modelNameTokens),
  );

  // Numeric reward keeps nearby versions ordered (e.g. 5.2 > 5.1 when 5.3 is missing).
  // Variant reward keeps suffix-sensitive families aligned (codex/haiku/opus).
  // Coverage penalty suppresses unrelated but superficially similar names.
  return (
    weightedTokenScore * TOKEN_PREFIX_REWARD_MULTIPLIER +
    numericMatchReward(evalSlug, modelId) +
    numericClosenessReward(evalSlug, modelId) +
    sameVariantReward(evalSlug, modelId, modelName) +
    bScaleRewardOrPenalty(evalSlug, modelId, modelName) +
    activeBRewardOrPenalty(evalSlug, modelId, modelName) +
    coverageRewardOrPenalty(evalSlug, modelId, modelName) +
    maxPrefix * CHAR_PREFIX_REWARD_SCALE -
    Math.abs(normalizedEval.length - normalizedModelBase.length) *
      LENGTH_GAP_PENALTY_SCALE
  );
}

function compareCandidates(
  left: LlmMatchCandidate,
  right: LlmMatchCandidate,
): number {
  if (left.score !== right.score) {
    return right.score - left.score;
  }
  return left.model_id.localeCompare(right.model_id);
}

type PreferredProviderScopedModels = {
  primary: ModelsDevModel[];
  fallback: ModelsDevModel[];
};

function uniqueModelCount(models: ModelsDevModel[]): number {
  return new Set(models.map((model) => model.model_id)).size;
}

function hasExactSlugFallbackCandidate(
  evalModel: ArtificialAnalysisModel,
  fallbackCandidates: LlmMatchCandidate[],
): boolean {
  const evalSlug =
    typeof evalModel.slug === "string" ? normalize(evalModel.slug) : "";
  if (evalSlug.length === 0) {
    return false;
  }
  return fallbackCandidates.some((candidate) => {
    const candidateSlug = normalize(splitBaseModelId(candidate.model_id));
    return candidateSlug.length > 0 && candidateSlug === evalSlug;
  });
}

function scopeToPreferredProviderModels(
  modelsDevModels: ModelsDevModel[],
): PreferredProviderScopedModels {
  const primary = modelsDevModels.filter(
    (modelStatsModel) =>
      modelStatsModel.provider_id === PRIMARY_PROVIDER_FILTER,
  );
  const fallback = modelsDevModels.filter((modelStatsModel) =>
    FALLBACK_PROVIDER_FILTERS.has(modelStatsModel.provider_id),
  );
  return { primary, fallback };
}

function selectPreferredCandidatesForEvalModel(
  evalModel: ArtificialAnalysisModel,
  scopedModels: PreferredProviderScopedModels,
): LlmMatchCandidate[] {
  const primaryCandidates = collectCandidatesForEvalModel(
    evalModel,
    scopedModels.primary,
  );
  const fallbackCandidates = collectCandidatesForEvalModel(
    evalModel,
    scopedModels.fallback,
  );
  if (primaryCandidates.length === 0) {
    return fallbackCandidates;
  }
  if (hasExactSlugFallbackCandidate(evalModel, fallbackCandidates)) {
    return fallbackCandidates;
  }
  return primaryCandidates;
}

function collectCandidatesForEvalModel(
  evalModel: ArtificialAnalysisModel,
  modelsDevModels: ModelsDevModel[],
): LlmMatchCandidate[] {
  const evalSlug = String(evalModel.slug ?? "");
  if (!evalSlug) {
    return [];
  }

  return modelsDevModels
    .map((modelStatsModel) => {
      const modelName =
        typeof modelStatsModel.model.name === "string"
          ? modelStatsModel.model.name
          : "";
      if (!hasFirstTokenMatch(evalSlug, modelStatsModel.model_id, modelName)) {
        return null;
      }
      const score = scoreCandidate(
        evalSlug,
        modelStatsModel.model_id,
        modelName,
      );
      if (score <= 0) {
        return null;
      }
      return {
        model_id: modelStatsModel.model_id,
        provider_id: modelStatsModel.provider_id,
        provider_name: modelStatsModel.provider_name,
        model_name: modelName || null,
        score,
      };
    })
    .filter((candidate): candidate is LlmMatchCandidate => candidate != null)
    .sort(compareCandidates);
}

function applyMaxMinHalfVoid<
  T extends { best_match: LlmMatchResult; candidates?: unknown[] },
>(models: T[]): { threshold: number | null; voided: number } {
  const scores = models
    .map((model) => model.best_match?.score)
    .filter((score): score is number => Number.isFinite(score))
    .sort((left, right) => left - right);
  if (scores.length === 0) {
    return { threshold: null, voided: 0 };
  }

  const minScore = scores[0] as number;
  const maxScore = scores.at(-1) as number;
  const threshold =
    minScore + (maxScore - minScore) * VOID_THRESHOLD_RANGE_RATIO;
  let voided = 0;
  for (const model of models) {
    const score = model.best_match?.score;
    if (score != null && score < threshold) {
      model.best_match = null;
      if ("candidates" in model && Array.isArray(model.candidates)) {
        model.candidates = [];
      }
      voided += 1;
    }
  }
  return { threshold, voided };
}

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

  const models: LlmMatchMappedModel[] = artificialAnalysisStats.models.map(
    (evalModel) => {
      const candidates = selectPreferredCandidatesForEvalModel(
        evalModel,
        scopedModels,
      ).slice(0, maxCandidates);
      return {
        artificial_analysis_slug:
          typeof evalModel.slug === "string" ? evalModel.slug : "",
        artificial_analysis_name:
          typeof evalModel.name === "string" ? evalModel.name : null,
        artificial_analysis_release_date:
          typeof evalModel.release_date === "string"
            ? evalModel.release_date
            : null,
        best_match: candidates[0] ?? null,
        candidates,
      };
    },
  );

  const voidStats = applyMaxMinHalfVoid(models);

  return {
    artificial_analysis_fetched_at_epoch_seconds:
      artificialAnalysisStats.fetched_at_epoch_seconds,
    models_dev_fetched_at_epoch_seconds:
      modelsDevStats.fetched_at_epoch_seconds,
    total_artificial_analysis_models: models.length,
    total_models_dev_models: totalScopedModels,
    max_candidates: maxCandidates,
    void_mode: "maxmin_half",
    void_threshold: voidStats.threshold,
    voided_count: voidStats.voided,
    models,
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

  const models: LlmMatchMappedModel[] = scrapedStats.data.map((row) => {
    const rowRecord = asRecord(row);
    const modelId =
      typeof rowRecord.model_id === "string" ? rowRecord.model_id : null;
    const slug = modelSlugFromModelId(modelId);

    const evalModel = {
      slug: slug ?? undefined,
      name: modelId ?? undefined,
      release_date: "",
    } as unknown as ArtificialAnalysisModel;

    const candidates = slug
      ? selectPreferredCandidatesForEvalModel(evalModel, scopedModels)
      : [];
    const topCandidates = candidates.slice(0, maxCandidates);

    return {
      artificial_analysis_slug: slug ?? "",
      artificial_analysis_name: modelId,
      artificial_analysis_release_date: null,
      best_match: topCandidates[0] ?? null,
      candidates: topCandidates,
    };
  });

  const preVoidMatchedCount = models.filter(
    (model) => model.best_match != null,
  ).length;
  const preVoidUnmatchedCount = models.length - preVoidMatchedCount;
  const voidStats = applyMaxMinHalfVoid(models);
  const matchedCount = models.filter(
    (model) => model.best_match != null,
  ).length;
  const unmatchedCount = models.length - matchedCount;

  return {
    scraped_fetched_at_epoch_seconds: scrapedStats.fetched_at_epoch_seconds,
    models_dev_fetched_at_epoch_seconds:
      modelsDevStats.fetched_at_epoch_seconds,
    total_scraped_models: scrapedStats.data.length,
    total_models_dev_models: totalScopedModels,
    max_candidates: maxCandidates,
    pre_void_matched_count: preVoidMatchedCount,
    pre_void_unmatched_count: preVoidUnmatchedCount,
    void_mode: "maxmin_half",
    void_threshold: voidStats.threshold,
    voided_count: voidStats.voided,
    matched_count: matchedCount,
    unmatched_count: unmatchedCount,
    models,
  };
}
