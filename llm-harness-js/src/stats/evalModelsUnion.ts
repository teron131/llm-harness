import { getEvalStats } from "./evalStats.js";
import { getModelStats } from "./modelStats.js";

type ModelStatsModel = Awaited<
  ReturnType<typeof getModelStats>
>["models"][number];
type EvalStatsModel = Awaited<
  ReturnType<typeof getEvalStats>
>["models"][number];

type BestMatch = {
  model_id: string;
  provider_id: string;
  provider_name: string;
  model_name: string | null;
  score: number;
} | null;

export type EvalModelMappingCandidate = NonNullable<BestMatch>;

export type EvalMappedModel = {
  eval_slug: string;
  eval_name: string | null;
  eval_release_date: string | null;
  best_match: BestMatch;
  candidates: EvalModelMappingCandidate[];
};

export type EvalModelMappingPayload = {
  eval_fetched_at_epoch_seconds: number;
  eval_status_code: number;
  models_dev_fetched_at_epoch_seconds: number;
  models_dev_status_code: number;
  total_eval_models: number;
  total_models_dev_models: number;
  provider_filter: string | null;
  max_candidates: number;
  void_mode: "maxmin_half";
  void_threshold: number | null;
  voided_count: number;
  models: EvalMappedModel[];
};

type EvalModelUnionRow = {
  eval_slug: string;
  eval_name: string | null;
  eval_release_date: string | null;
  best_match: BestMatch;
  eval: EvalStatsModel;
  models_dev: ModelStatsModel | null;
  union: Record<string, unknown>;
};

export type EvalModelsUnionPayload = {
  eval_fetched_at_epoch_seconds: number;
  eval_status_code: number;
  models_dev_fetched_at_epoch_seconds: number;
  models_dev_status_code: number;
  total_eval_models: number;
  total_models_dev_models: number;
  provider_filter: string;
  merge_mode: "union_only";
  void_mode: "maxmin_half";
  void_threshold: number | null;
  voided_count: number;
  total_union_models: number;
  models: Array<Record<string, unknown>>;
};

export type EvalModelsUnionOptions = {
  providerFilter?: string;
};

export type EvalModelMappingOptions = {
  providerFilter?: string | null;
  maxCandidates?: number;
};

const TOKEN_PREFIX_WEIGHTS = [4, 3, 2, 1] as const;
const DEFAULT_MAX_CANDIDATES = 5;
const TOKEN_PREFIX_MULTIPLIER = 2;
const NUMERIC_EXACT_MATCH_REWARD = 2;
const NUMERIC_CLOSENESS_SCALE = 0.1;
const NUMERIC_ALL_EQUAL_REWARD = 0.2;
const VARIANT_SUFFIX_REWARD = 2;
const COVERAGE_EXACT_REWARD = 2;
const COVERAGE_MISSING_BASE_PENALTY = 1;
const CHAR_PREFIX_REWARD_SCALE = 0.01;
const LENGTH_GAP_PENALTY_SCALE = 0.001;

function normalize(value: string): string {
  return value
    .toLowerCase()
    .replace(/[._:\s]+/g, "-")
    .replace(/[^a-z0-9/-]+/g, "")
    .replace(/-+/g, "-")
    .replace(/^[-/]+|[-/]+$/g, "");
}

function splitBaseModelId(modelId: string): string {
  const parts = modelId.split("/");
  return parts[parts.length - 1] ?? modelId;
}

function splitTokens(value: string): string[] {
  return normalize(value).split("-").filter(Boolean);
}

function isNumericToken(token: string | undefined): boolean {
  return Boolean(token && /^\d+$/.test(token));
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

function numericMatchBonus(evalSlug: string, modelId: string): number {
  const evalTokens = splitTokens(evalSlug);
  const modelTokens = splitTokens(splitBaseModelId(modelId));
  const maxLength = Math.min(evalTokens.length, modelTokens.length);
  for (let index = 0; index < maxLength; index += 1) {
    const evalToken = evalTokens[index];
    const modelToken = modelTokens[index];
    if (isNumericToken(evalToken) && isNumericToken(modelToken)) {
      return evalToken === modelToken ? NUMERIC_EXACT_MATCH_REWARD : 0;
    }
  }
  return 0;
}

function numericClosenessBonus(evalSlug: string, modelId: string): number {
  const evalNumbers = splitTokens(evalSlug)
    .filter((token) => isNumericToken(token))
    .map((token) => Number(token));
  const modelNumbers = splitTokens(splitBaseModelId(modelId))
    .filter((token) => isNumericToken(token))
    .map((token) => Number(token));

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
    return NUMERIC_CLOSENESS_SCALE / (1 + Math.abs(evalValue - modelValue));
  }
  return NUMERIC_ALL_EQUAL_REWARD;
}

function sameVariantBonus(
  evalSlug: string,
  modelId: string,
  modelName: string,
): number {
  const evalTokens = splitTokens(evalSlug);
  const modelBaseTokens = splitTokens(splitBaseModelId(modelId));
  const modelNameTokens = splitTokens(modelName);
  const evalLastToken = evalTokens[evalTokens.length - 1];
  if (!evalLastToken || isNumericToken(evalLastToken)) {
    return 0;
  }
  const baseLastToken = modelBaseTokens[modelBaseTokens.length - 1];
  const nameLastToken = modelNameTokens[modelNameTokens.length - 1];
  if (evalLastToken === baseLastToken || evalLastToken === nameLastToken) {
    return VARIANT_SUFFIX_REWARD;
  }
  return 0;
}

function setCoverageBonus(
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

  const weightedTokenScore = Math.max(
    weightedTokenPrefixScore(evalTokens, modelBaseTokens),
    weightedTokenPrefixScore(evalTokens, modelNameTokens),
  );

  return (
    weightedTokenScore * TOKEN_PREFIX_MULTIPLIER +
    numericMatchBonus(evalSlug, modelId) +
    numericClosenessBonus(evalSlug, modelId) +
    sameVariantBonus(evalSlug, modelId, modelName) +
    setCoverageBonus(evalSlug, modelId, modelName) +
    maxPrefix * CHAR_PREFIX_REWARD_SCALE -
    Math.abs(normalizedEval.length - normalizedModelBase.length) *
      LENGTH_GAP_PENALTY_SCALE
  );
}

function compareByScore(left: BestMatch, right: BestMatch): number {
  if (!left || !right) {
    return 0;
  }
  if (left.score !== right.score) {
    return right.score - left.score;
  }
  return left.model_id.localeCompare(right.model_id);
}

function bestMatchForEvalModel(
  evalModel: EvalStatsModel,
  modelStatsModels: ModelStatsModel[],
  providerFilter: string,
): BestMatch {
  const evalSlug = String(evalModel.slug ?? "");
  if (!evalSlug) {
    return null;
  }

  const candidates: NonNullable<BestMatch>[] = modelStatsModels
    .filter((modelStatsModel) => modelStatsModel.provider_id === providerFilter)
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
    .filter(
      (candidate): candidate is NonNullable<BestMatch> => candidate != null,
    );

  if (candidates.length === 0) {
    return null;
  }
  candidates.sort(compareByScore);
  return candidates[0] ?? null;
}

function topCandidatesForEvalModel(
  evalModel: EvalStatsModel,
  modelStatsModels: ModelStatsModel[],
  providerFilter: string,
  maxCandidates: number,
): EvalModelMappingCandidate[] {
  const evalSlug = String(evalModel.slug ?? "");
  if (!evalSlug) {
    return [];
  }

  return modelStatsModels
    .filter((modelStatsModel) => modelStatsModel.provider_id === providerFilter)
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
    .filter(
      (candidate): candidate is EvalModelMappingCandidate => candidate != null,
    )
    .sort(compareByScore)
    .slice(0, maxCandidates);
}

function applyMaxMinHalfVoid(models: EvalModelUnionRow[]): {
  threshold: number | null;
  voided: number;
} {
  const scores = models
    .map((model) => model.best_match?.score)
    .filter((score): score is number => Number.isFinite(score))
    .sort((left, right) => left - right);
  if (scores.length === 0) {
    return { threshold: null, voided: 0 };
  }

  const min = scores[0] as number;
  const max = scores[scores.length - 1] as number;
  const threshold = min + (max - min) / 2;
  let voided = 0;
  for (const model of models) {
    const score = model.best_match?.score;
    if (score != null && score < threshold) {
      model.best_match = null;
      voided += 1;
    }
  }
  return { threshold, voided };
}

function applyMaxMinHalfVoidForMapping(models: EvalMappedModel[]): {
  threshold: number | null;
  voided: number;
} {
  const scores = models
    .map((model) => model.best_match?.score)
    .filter((score): score is number => Number.isFinite(score))
    .sort((left, right) => left - right);
  if (scores.length === 0) {
    return { threshold: null, voided: 0 };
  }

  const min = scores[0] as number;
  const max = scores[scores.length - 1] as number;
  const threshold = min + (max - min) / 2;
  let voided = 0;
  for (const model of models) {
    const score = model.best_match?.score;
    if (score != null && score < threshold) {
      model.best_match = null;
      model.candidates = [];
      voided += 1;
    }
  }
  return { threshold, voided };
}

export async function getEvalModelsUnion(
  options: EvalModelsUnionOptions = {},
): Promise<EvalModelsUnionPayload> {
  const providerFilter = options.providerFilter?.trim() || "openrouter";
  const evalStats = await getEvalStats();
  const modelStats = await getModelStats();

  const scopedModelStatsModels = modelStats.models.filter(
    (model) => model.provider_id === providerFilter,
  );

  const rows: EvalModelUnionRow[] = evalStats.models.map((evalModel) => {
    const bestMatch = bestMatchForEvalModel(
      evalModel,
      scopedModelStatsModels,
      providerFilter,
    );
    const matchedModelStats = bestMatch
      ? (scopedModelStatsModels.find(
          (model) => model.model_id === bestMatch.model_id,
        ) ?? null)
      : null;

    return {
      eval_slug: typeof evalModel.slug === "string" ? evalModel.slug : "",
      eval_name: typeof evalModel.name === "string" ? evalModel.name : null,
      eval_release_date:
        typeof evalModel.release_date === "string"
          ? evalModel.release_date
          : null,
      best_match: bestMatch,
      eval: evalModel,
      models_dev: matchedModelStats,
      union: {
        ...(matchedModelStats?.model ?? {}),
        ...(evalModel ?? {}),
      },
    };
  });

  const voidStats = applyMaxMinHalfVoid(rows);
  const unions = rows
    .filter((row) => row.best_match != null)
    .map((row) => row.union);

  return {
    eval_fetched_at_epoch_seconds: evalStats.fetched_at_epoch_seconds,
    eval_status_code: evalStats.status_code,
    models_dev_fetched_at_epoch_seconds: modelStats.fetched_at_epoch_seconds,
    models_dev_status_code: modelStats.status_code,
    total_eval_models: evalStats.models.length,
    total_models_dev_models: scopedModelStatsModels.length,
    provider_filter: providerFilter,
    merge_mode: "union_only",
    void_mode: "maxmin_half",
    void_threshold: voidStats.threshold,
    voided_count: voidStats.voided,
    total_union_models: unions.length,
    models: unions,
  };
}

export async function getEvalModelMapping(
  options: EvalModelMappingOptions = {},
): Promise<EvalModelMappingPayload> {
  const providerFilter = options.providerFilter?.trim() || "openrouter";
  const maxCandidates = options.maxCandidates ?? DEFAULT_MAX_CANDIDATES;
  const evalStats = await getEvalStats();
  const modelStats = await getModelStats();

  const models: EvalMappedModel[] = evalStats.models.map((evalModel) => {
    const candidates = topCandidatesForEvalModel(
      evalModel,
      modelStats.models,
      providerFilter,
      maxCandidates,
    );
    return {
      eval_slug: typeof evalModel.slug === "string" ? evalModel.slug : "",
      eval_name: typeof evalModel.name === "string" ? evalModel.name : null,
      eval_release_date:
        typeof evalModel.release_date === "string"
          ? evalModel.release_date
          : null,
      best_match: candidates[0] ?? null,
      candidates,
    };
  });

  const voidStats = applyMaxMinHalfVoidForMapping(models);

  return {
    eval_fetched_at_epoch_seconds: evalStats.fetched_at_epoch_seconds,
    eval_status_code: evalStats.status_code,
    models_dev_fetched_at_epoch_seconds: modelStats.fetched_at_epoch_seconds,
    models_dev_status_code: modelStats.status_code,
    total_eval_models: models.length,
    total_models_dev_models: modelStats.models.length,
    provider_filter: providerFilter,
    max_candidates: maxCandidates,
    void_mode: "maxmin_half",
    void_threshold: voidStats.threshold,
    voided_count: voidStats.voided,
    models,
  };
}
