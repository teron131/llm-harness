import { add, mean, multiply } from "mathjs";

const MODELS_URL = "https://artificialanalysis.ai/api/v2/data/llms/models";
const LOOKBACK_DAYS = 365;
const REQUEST_TIMEOUT_MS = 30_000;
const BENCHMARK_KEYS = [
  "hle",
  "terminalbench_hard",
  "lcr",
  "ifbench",
  "scicode",
] as const;

const SCORE_WEIGHTS = {
  intelligence: 0.3,
  benchmark_bias: 0.3,
  price: 0.2,
  speed: 0.2,
} as const;

type NumberOrNull = number | null;
type ModelCreator = {
  name?: string;
  slug?: string;
};

type Evaluations = {
  artificial_analysis_intelligence_index?: number | null;
  artificial_analysis_coding_index?: number | null;
  hle?: number | null;
  terminalbench_hard?: number | null;
  lcr?: number | null;
  ifbench?: number | null;
  scicode?: number | null;
  [key: string]: unknown;
};

type Pricing = {
  price_1m_blended_3_to_1?: number | null;
  price_1m_input_tokens?: number | null;
  price_1m_output_tokens?: number | null;
  [key: string]: unknown;
};

type BaseModel = {
  name?: string;
  slug?: string;
  release_date?: string;
  model_creator?: ModelCreator;
  evaluations?: Evaluations;
  pricing?: Pricing;
  median_output_tokens_per_second?: number | null;
  median_time_to_first_token_seconds?: number | null;
  median_time_to_first_answer_token?: number | null;
  [key: string]: unknown;
};

type Scores = {
  overall_score: NumberOrNull;
  intelligence_score: NumberOrNull;
  benchmark_bias_score: NumberOrNull;
  price_score: NumberOrNull;
  speed_score: NumberOrNull;
};

type Percentiles = {
  overall_percentile: NumberOrNull;
  intelligence_percentile: NumberOrNull;
  speed_percentile: NumberOrNull;
  price_percentile: NumberOrNull;
};

type ScoredModel = BaseModel & { scores: Scores };

export type ArtificialAnalysisEnrichedModel = BaseModel & {
  scores: Scores;
  percentiles: Percentiles;
};

type SourcePayload = {
  fetched_at_epoch_seconds: number | null;
  status_code: number | null;
  models: BaseModel[];
};

export type ArtificialAnalysisOutputPayload = {
  fetched_at_epoch_seconds: number | null;
  status_code: number | null;
  models: ArtificialAnalysisEnrichedModel[];
};

export type ArtificialAnalysisOptions = { apiKey?: string };

function finiteNumbers(values: unknown[]): number[] {
  return values
    .filter((value) => value != null)
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
}

function meanOfFinite(values: unknown[]): NumberOrNull {
  const numbers = finiteNumbers(values);
  if (numbers.length === 0) {
    return null;
  }
  return Number(mean(numbers));
}

function isPositiveFinite(value: unknown): boolean {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) && numericValue > 0;
}

function signedLog(value: unknown, invert = false): NumberOrNull {
  if (!isPositiveFinite(value)) {
    return null;
  }
  const numericValue = Number(value);
  return invert ? -Math.log(numericValue) : Math.log(numericValue);
}

function weightedMean(
  pairs: Array<{ value: NumberOrNull; weight: number }>,
): NumberOrNull {
  const validPairs = pairs.filter(
    (pair) =>
      pair.value != null &&
      Number.isFinite(pair.value) &&
      Number.isFinite(pair.weight) &&
      pair.weight > 0,
  );
  if (validPairs.length === 0) {
    return null;
  }
  const weightedSum = validPairs.reduce(
    (sum, pair) => sum + (pair.value as number) * pair.weight,
    0,
  );
  const weightSum = validPairs.reduce((sum, pair) => sum + pair.weight, 0);
  if (weightSum === 0) {
    return null;
  }
  return weightedSum / weightSum;
}

function percentileRank(values: unknown[], value: unknown): NumberOrNull {
  if (value == null) {
    return null;
  }
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return null;
  }
  const finiteValues = finiteNumbers(values);
  if (finiteValues.length === 0) {
    return null;
  }
  const lessOrEqualCount = finiteValues.filter(
    (item) => item <= numericValue,
  ).length;
  return (lessOrEqualCount / finiteValues.length) * 100;
}

function removeIds<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map((item) => removeIds(item)) as T;
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([key]) => key !== "id")
        .map(([key, child]) => [key, removeIds(child)]),
    ) as T;
  }
  return value;
}

function computeScores(filteredModels: BaseModel[]): ScoredModel[] {
  return filteredModels.map((model) => {
    const intelligence = Number(
      model.evaluations?.artificial_analysis_intelligence_index,
    );
    const coding = Number(model.evaluations?.artificial_analysis_coding_index);
    const blendedPrice = model.pricing?.price_1m_blended_3_to_1;
    const ttfa = model.median_time_to_first_answer_token;
    const tps = model.median_output_tokens_per_second;

    const intelligenceRaw =
      Number.isFinite(intelligence) && Number.isFinite(coding)
        ? Number(add(multiply(2, intelligence), coding))
        : null;
    const priceRaw = signedLog(blendedPrice, true);
    const ttfaRaw = signedLog(ttfa, true);
    const tpsRaw = signedLog(tps);
    const intelligenceScore = intelligenceRaw;
    const benchmarkBiasScore = meanOfFinite(
      BENCHMARK_KEYS.map((key) => {
        if (!isPositiveFinite(model.evaluations?.[key])) {
          return null;
        }
        return Number(model.evaluations?.[key]);
      }),
    );
    const priceScore = priceRaw;
    const speedScore = meanOfFinite([ttfaRaw, tpsRaw]);

    return {
      ...model,
      scores: {
        overall_score: weightedMean([
          { value: intelligenceScore, weight: SCORE_WEIGHTS.intelligence },
          { value: benchmarkBiasScore, weight: SCORE_WEIGHTS.benchmark_bias },
          { value: priceScore, weight: SCORE_WEIGHTS.price },
          { value: speedScore, weight: SCORE_WEIGHTS.speed },
        ]),
        intelligence_score: intelligenceScore,
        benchmark_bias_score: benchmarkBiasScore,
        price_score: priceScore,
        speed_score: speedScore,
      },
    };
  });
}

function rankAndEnrichModels(
  models: BaseModel[],
  cutoffDate: string,
): ArtificialAnalysisEnrichedModel[] {
  const filteredModels = models.filter((model) => {
    return (
      (model.release_date ?? "") >= cutoffDate &&
      isPositiveFinite(model.pricing?.price_1m_blended_3_to_1) &&
      isPositiveFinite(model.pricing?.price_1m_input_tokens) &&
      isPositiveFinite(model.pricing?.price_1m_output_tokens) &&
      isPositiveFinite(model.median_time_to_first_answer_token) &&
      isPositiveFinite(model.median_output_tokens_per_second)
    );
  });

  const scoredModels = computeScores(filteredModels);
  const ranked = scoredModels
    .filter((model) => Number.isFinite(model.scores.overall_score))
    .sort(
      (left, right) =>
        (right.scores.overall_score ?? Number.NEGATIVE_INFINITY) -
        (left.scores.overall_score ?? Number.NEGATIVE_INFINITY),
    );

  const overallValues = ranked.map((model) => model.scores.overall_score);
  const intelligenceValues = ranked.map(
    (model) => model.scores.intelligence_score,
  );
  const speedValues = ranked.map((model) => model.scores.speed_score);
  const priceValues = ranked.map((model) => model.scores.price_score);

  return ranked.map((model) => ({
    ...model,
    percentiles: {
      overall_percentile: percentileRank(
        overallValues,
        model.scores.overall_score,
      ),
      intelligence_percentile: percentileRank(
        intelligenceValues,
        model.scores.intelligence_score,
      ),
      speed_percentile: percentileRank(speedValues, model.scores.speed_score),
      price_percentile: percentileRank(priceValues, model.scores.price_score),
    },
  }));
}

async function fetchModels(apiKey: string | undefined): Promise<SourcePayload> {
  if (!apiKey) {
    throw new Error("Missing ARTIFICIALANALYSIS_API_KEY.");
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const response = await fetch(MODELS_URL, {
    headers: { "x-api-key": apiKey },
    signal: controller.signal,
  });
  clearTimeout(timeout);

  if (!response.ok) {
    throw new Error(`Artificial Analysis request failed: ${response.status}`);
  }

  const payload = (await response.json()) as { data: BaseModel[] };
  const sourcePayload: SourcePayload = {
    fetched_at_epoch_seconds: Math.floor(Date.now() / 1000),
    status_code: response.status,
    models: payload.data.map((model) => removeIds(model)),
  };
  return sourcePayload;
}

export async function getArtificialAnalysisStats(
  options: ArtificialAnalysisOptions = {},
): Promise<ArtificialAnalysisOutputPayload> {
  try {
    const apiKey = options.apiKey ?? process.env.ARTIFICIALANALYSIS_API_KEY;
    const sourcePayload = await fetchModels(apiKey);

    const cutoffDate = new Date(
      Date.now() - LOOKBACK_DAYS * 24 * 60 * 60 * 1000,
    )
      .toISOString()
      .slice(0, 10);

    return {
      fetched_at_epoch_seconds: sourcePayload.fetched_at_epoch_seconds,
      status_code: sourcePayload.status_code,
      models: rankAndEnrichModels(sourcePayload.models, cutoffDate),
    };
  } catch {
    return {
      fetched_at_epoch_seconds: null,
      status_code: null,
      models: [],
    };
  }
}
