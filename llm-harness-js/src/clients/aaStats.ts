import { add, divide, log, mean, multiply } from "mathjs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const MODELS_URL = "https://artificialanalysis.ai/api/v2/data/llms/models";
const CACHE_PATH = resolve(".cache/aa_models.json");
const OUTPUT_PATH = resolve(".cache/aa_output.json");
const LOOKBACK_DAYS = 365;
const REQUEST_TIMEOUT_MS = 30_000;
const DEFAULT_CACHE_TTL_SECONDS = 60 * 60 * 12;

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

export type AaEnrichedModel = BaseModel & {
  scores: Scores;
  percentiles: Percentiles;
};

type CachePayload = {
  fetched_at_epoch_seconds: number;
  status_code: number;
  models: BaseModel[];
};

export type AaOutputPayload = {
  fetched_at_epoch_seconds: number;
  status_code: number;
  models: AaEnrichedModel[];
};

function inverseLog1p(value: number): number {
  return Number(divide(1, log(add(1, value))));
}

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

function meanOfPositive(values: unknown[]): NumberOrNull {
  const numbers = finiteNumbers(values).filter((value) => value > 0);
  if (numbers.length === 0) {
    return null;
  }
  return Number(mean(numbers));
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

function calculateScores(model: BaseModel): Scores {
  const evaluations = model.evaluations ?? {};
  const pricing = model.pricing ?? {};

  const intelligenceIndex = Number(
    evaluations.artificial_analysis_intelligence_index,
  );
  const codingIndex = Number(evaluations.artificial_analysis_coding_index);
  const blendedPrice = Number(pricing.price_1m_blended_3_to_1);
  const ttfa = Number(model.median_time_to_first_answer_token);
  const tps = Number(model.median_output_tokens_per_second);

  const intelligenceScore =
    Number.isFinite(intelligenceIndex) && Number.isFinite(codingIndex)
      ? Number(add(multiply(2, intelligenceIndex), codingIndex))
      : null;

  const benchmarkBiasScore = meanOfPositive([
    evaluations.hle,
    evaluations.terminalbench_hard,
    evaluations.lcr,
    evaluations.ifbench,
    evaluations.scicode,
  ]);

  const priceScore =
    Number.isFinite(blendedPrice) && blendedPrice > 0
      ? inverseLog1p(blendedPrice)
      : null;

  const speedScore =
    Number.isFinite(ttfa) && Number.isFinite(tps) && ttfa > 0 && tps > 0
      ? Number(add(inverseLog1p(ttfa), log(tps)))
      : null;

  return {
    overall_score: meanOfFinite([
      intelligenceScore,
      benchmarkBiasScore,
      priceScore,
      speedScore,
    ]),
    intelligence_score: intelligenceScore,
    benchmark_bias_score: benchmarkBiasScore,
    price_score: priceScore,
    speed_score: speedScore,
  };
}

function rankAndEnrichModels(
  models: BaseModel[],
  cutoffDate: string,
): AaEnrichedModel[] {
  const filteredModels = models.filter((model) => {
    const blendedPrice = Number(model.pricing?.price_1m_blended_3_to_1);
    const inputPrice = Number(model.pricing?.price_1m_input_tokens);
    const outputPrice = Number(model.pricing?.price_1m_output_tokens);
    const ttfa = Number(model.median_time_to_first_answer_token);
    const tps = Number(model.median_output_tokens_per_second);

    return (
      (model.release_date ?? "") >= cutoffDate &&
      Number.isFinite(blendedPrice) &&
      Number.isFinite(inputPrice) &&
      Number.isFinite(outputPrice) &&
      Number.isFinite(ttfa) &&
      Number.isFinite(tps) &&
      blendedPrice > 0 &&
      inputPrice > 0 &&
      outputPrice > 0 &&
      ttfa > 0 &&
      tps > 0
    );
  });

  const scoredModels = filteredModels.map((model) => ({
    ...model,
    scores: calculateScores(model),
  }));

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

async function loadCache(): Promise<CachePayload> {
  const content = await readFile(CACHE_PATH, "utf-8");
  return JSON.parse(content) as CachePayload;
}

async function writeJson(path: string, payload: unknown): Promise<void> {
  await mkdir(resolve(".cache"), { recursive: true });
  await writeFile(path, JSON.stringify(payload, null, 2), "utf-8");
}

async function fetchAndCacheModels(
  apiKey: string | undefined,
): Promise<CachePayload> {
  if (!apiKey) {
    throw new Error(
      "Missing ARTIFICIALANALYSIS_API_KEY for refresh. Set it or use existing cache.",
    );
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
  const cachePayload: CachePayload = {
    fetched_at_epoch_seconds: Math.floor(Date.now() / 1000),
    status_code: response.status,
    models: payload.data.map((model) => removeIds(model)),
  };
  await writeJson(CACHE_PATH, cachePayload);
  return cachePayload;
}

export async function getAaStatsJson(): Promise<AaOutputPayload> {
  const apiKey = process.env.ARTIFICIALANALYSIS_API_KEY;

  const refreshCache = process.env.AA_REFRESH === "1";
  const cacheTtlSeconds = Number(
    process.env.AA_CACHE_TTL_SECONDS ?? DEFAULT_CACHE_TTL_SECONDS,
  );

  let cachePayload: CachePayload;
  if (!refreshCache) {
    try {
      const cached = await loadCache();
      const ageSeconds =
        Math.floor(Date.now() / 1000) - cached.fetched_at_epoch_seconds;
      if (ageSeconds <= cacheTtlSeconds) {
        cachePayload = cached;
      } else {
        cachePayload = await fetchAndCacheModels(apiKey);
      }
    } catch {
      cachePayload = await fetchAndCacheModels(apiKey);
    }
  } else {
    cachePayload = await fetchAndCacheModels(apiKey);
  }

  const cutoffDate = new Date(Date.now() - LOOKBACK_DAYS * 24 * 60 * 60 * 1000)
    .toISOString()
    .slice(0, 10);

  const outputPayload: AaOutputPayload = {
    fetched_at_epoch_seconds: cachePayload.fetched_at_epoch_seconds,
    status_code: cachePayload.status_code,
    models: rankAndEnrichModels(cachePayload.models, cutoffDate),
  };

  await writeJson(OUTPUT_PATH, outputPayload);
  return outputPayload;
}
