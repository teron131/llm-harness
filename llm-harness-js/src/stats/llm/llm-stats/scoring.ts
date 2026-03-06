import { percentileRank } from "../../utils.js";
import { asFiniteNumber, asRecord, type JsonObject } from "../shared.js";

import { type ModelStatsSelectedModel } from "./types.js";

const INTELLIGENCE_BENCHMARK_KEYS = [
  "omniscience_accuracy",
  "hle",
  "lcr",
  "scicode",
] as const;
const AGENTIC_BENCHMARK_KEYS = [
  "omniscience_nonhallucination_rate",
  "gdpval_normalized",
  "ifbench",
  "terminalbench_hard",
] as const;
const DEFAULT_SPEED_OUTPUT_TOKEN_ANCHORS = [
  200, 500, 1_000, 2_000, 8_000,
] as const;
const SPEED_OUTPUT_TOKEN_RANGE_MIN = 200;
const SPEED_OUTPUT_TOKEN_RANGE_MAX = 8_000;
const SPEED_ANCHOR_QUANTILES = [0.25, 0.5, 0.75] as const;
const WEIGHTED_PRICE_INPUT_RATIO = 0.75;
const WEIGHTED_PRICE_OUTPUT_RATIO = 0.25;

function meanOfFinite(values: Array<number | null>): number | null {
  const finiteValues = values.filter(
    (value): value is number => value != null && Number.isFinite(value),
  );
  if (finiteValues.length === 0) {
    return null;
  }
  const total = finiteValues.reduce((sum, value) => sum + value, 0);
  return total / finiteValues.length;
}

function metricValue(model: JsonObject, key: string): number | null {
  const intelligence = asRecord(model.intelligence);
  const evaluations = asRecord(model.evaluations);
  if (key === "omniscience_nonhallucination_rate") {
    const nonhallucinationRate = asFiniteNumber(
      intelligence.omniscience_nonhallucination_rate,
    );
    if (nonhallucinationRate != null) {
      return nonhallucinationRate;
    }
    const nonhallucinationRateFromLegacyKey = asFiniteNumber(
      intelligence.omniscience_hallucination_rate,
    );
    return nonhallucinationRateFromLegacyKey;
  }
  return (
    asFiniteNumber(intelligence[key]) ??
    asFiniteNumber(evaluations[key]) ??
    null
  );
}

export function blendedPriceValue(costLike: unknown): number | null {
  const cost = asRecord(costLike);
  const inputCost = asFiniteNumber(cost.input);
  const outputCost = asFiniteNumber(cost.output);
  const weightedInputCost = asFiniteNumber(cost.weighted_input);
  const weightedOutputCost = asFiniteNumber(cost.weighted_output);
  const cacheReadCost = asFiniteNumber(cost.cache_read);
  const cacheWriteCost = asFiniteNumber(cost.cache_write);
  if (
    inputCost == null ||
    outputCost == null ||
    inputCost <= 0 ||
    outputCost <= 0
  ) {
    return null;
  }
  if (weightedInputCost != null || weightedOutputCost != null) {
    const effectiveInputCost =
      weightedInputCost != null ? weightedInputCost : inputCost;
    const effectiveOutputCost =
      weightedOutputCost != null ? weightedOutputCost : outputCost;
    return (
      WEIGHTED_PRICE_INPUT_RATIO * effectiveInputCost +
      WEIGHTED_PRICE_OUTPUT_RATIO * effectiveOutputCost
    );
  }

  const cacheWeightedInput = cacheReadCost != null ? cacheReadCost : inputCost;
  const cacheWeightedOutput =
    cacheWriteCost != null
      ? 0.1 * cacheWriteCost + 0.9 * outputCost
      : outputCost;
  const baseProxy =
    WEIGHTED_PRICE_INPUT_RATIO *
      (WEIGHTED_PRICE_INPUT_RATIO * cacheWeightedInput +
        WEIGHTED_PRICE_OUTPUT_RATIO * inputCost) +
    WEIGHTED_PRICE_OUTPUT_RATIO * cacheWeightedOutput;

  const over200kCost = asRecord(cost.context_over_200k);
  const over200kInput = asFiniteNumber(over200kCost.input);
  const over200kOutput = asFiniteNumber(over200kCost.output);
  const over200kCacheRead = asFiniteNumber(over200kCost.cache_read);
  const over200kCacheWrite = asFiniteNumber(over200kCost.cache_write);
  if (
    over200kInput == null ||
    over200kOutput == null ||
    over200kInput <= 0 ||
    over200kOutput <= 0
  ) {
    return baseProxy;
  }

  const over200kInputWeighted =
    over200kCacheRead != null ? over200kCacheRead : over200kInput;
  const over200kOutputWeighted =
    over200kCacheWrite != null
      ? 0.1 * over200kCacheWrite + 0.9 * over200kOutput
      : over200kOutput;
  const over200kProxy =
    WEIGHTED_PRICE_INPUT_RATIO *
      (WEIGHTED_PRICE_INPUT_RATIO * over200kInputWeighted +
        WEIGHTED_PRICE_OUTPUT_RATIO * over200kInput) +
    WEIGHTED_PRICE_OUTPUT_RATIO * over200kOutputWeighted;

  return 0.95 * baseProxy + 0.05 * over200kProxy;
}

function quantileFromSorted(values: number[], quantile: number): number | null {
  if (values.length === 0) {
    return null;
  }
  if (values.length === 1) {
    return values[0] ?? null;
  }
  const clampedQuantile = Math.min(1, Math.max(0, quantile));
  const index = (values.length - 1) * clampedQuantile;
  const lowerIndex = Math.floor(index);
  const upperIndex = Math.ceil(index);
  if (lowerIndex === upperIndex) {
    return values[lowerIndex] ?? null;
  }
  const lowerValue = values[lowerIndex];
  const upperValue = values[upperIndex];
  if (lowerValue == null || upperValue == null) {
    return null;
  }
  const ratio = index - lowerIndex;
  return lowerValue + (upperValue - lowerValue) * ratio;
}

export function deriveSpeedOutputTokenAnchors(
  openRouterSpeedById: Map<string, JsonObject>,
): number[] {
  const impliedTokenUsages = Array.from(openRouterSpeedById.values())
    .map((speed) => {
      const throughputTokensPerSecond = asFiniteNumber(
        speed.throughput_tokens_per_second_median,
      );
      const latencySeconds = asFiniteNumber(speed.latency_seconds_median);
      const e2eLatencySeconds = asFiniteNumber(
        speed.e2e_latency_seconds_median,
      );
      if (
        throughputTokensPerSecond == null ||
        throughputTokensPerSecond <= 0 ||
        latencySeconds == null ||
        e2eLatencySeconds == null
      ) {
        return null;
      }
      const generationSeconds = e2eLatencySeconds - latencySeconds;
      if (generationSeconds <= 0) {
        return null;
      }
      return generationSeconds * throughputTokensPerSecond;
    })
    .filter((value): value is number => value != null && Number.isFinite(value))
    .sort((left, right) => left - right);

  if (impliedTokenUsages.length === 0) {
    return [...DEFAULT_SPEED_OUTPUT_TOKEN_ANCHORS];
  }

  const q0 = impliedTokenUsages[0] ?? null;
  const [q1, q2, q3] = SPEED_ANCHOR_QUANTILES.map((quantile) =>
    quantileFromSorted(impliedTokenUsages, quantile),
  );
  const q4 = impliedTokenUsages[impliedTokenUsages.length - 1] ?? null;
  const numericQuantileAnchors = [q0, q1, q2, q3, q4].filter(
    (value): value is number => value != null && Number.isFinite(value),
  );
  if (numericQuantileAnchors.length !== 5) {
    return [...DEFAULT_SPEED_OUTPUT_TOKEN_ANCHORS];
  }

  const sourceMin = numericQuantileAnchors[0] as number;
  const sourceMax = numericQuantileAnchors[
    numericQuantileAnchors.length - 1
  ] as number;
  if (!(sourceMax > sourceMin)) {
    return [...DEFAULT_SPEED_OUTPUT_TOKEN_ANCHORS];
  }

  return numericQuantileAnchors.map((anchor) => {
    const normalized = (anchor - sourceMin) / (sourceMax - sourceMin);
    const mapped =
      SPEED_OUTPUT_TOKEN_RANGE_MIN +
      normalized *
        (SPEED_OUTPUT_TOKEN_RANGE_MAX - SPEED_OUTPUT_TOKEN_RANGE_MIN);
    return Math.round(mapped);
  });
}

export function buildScores(
  model: JsonObject,
  cost: unknown,
  speed: JsonObject,
  speedOutputTokenAnchors: number[],
): unknown {
  const intelligenceIndex =
    metricValue(model, "intelligence_index") ??
    metricValue(model, "artificial_analysis_intelligence_index");
  const agenticIndex =
    metricValue(model, "agentic_index") ??
    metricValue(model, "artificial_analysis_agentic_index");
  const intelligenceBenchmarkMean = meanOfFinite(
    INTELLIGENCE_BENCHMARK_KEYS.map((key) => metricValue(model, key)),
  );
  const agenticBenchmarkMean = meanOfFinite(
    AGENTIC_BENCHMARK_KEYS.map((key) => metricValue(model, key)),
  );
  const intelligenceScore =
    intelligenceIndex != null && intelligenceBenchmarkMean != null
      ? (intelligenceIndex + intelligenceBenchmarkMean) / 2
      : null;
  const agenticScore =
    agenticIndex != null && agenticBenchmarkMean != null
      ? (agenticIndex + agenticBenchmarkMean) / 2
      : null;
  const blendedPrice = blendedPriceValue(cost);
  const latencySeconds = asFiniteNumber(speed.latency_seconds_median);
  const throughputTokensPerSecond = asFiniteNumber(
    speed.throughput_tokens_per_second_median,
  );
  const e2eLatencySeconds = asFiniteNumber(speed.e2e_latency_seconds_median);
  const priceScore =
    blendedPrice != null && blendedPrice > 0 ? 1 / blendedPrice : null;
  const imaginedSpeedScore = meanOfFinite(
    speedOutputTokenAnchors.map((targetTokens) =>
      latencySeconds != null &&
      throughputTokensPerSecond != null &&
      throughputTokensPerSecond > 0
        ? targetTokens /
          (latencySeconds + targetTokens / throughputTokensPerSecond)
        : null,
    ),
  );
  const sortedAnchors = [...speedOutputTokenAnchors].sort(
    (left, right) => left - right,
  );
  const representativeTargetTokens = quantileFromSorted(sortedAnchors, 0.5);
  const observedE2eSpeedScore =
    representativeTargetTokens != null &&
    e2eLatencySeconds != null &&
    e2eLatencySeconds > 0
      ? representativeTargetTokens / e2eLatencySeconds
      : null;
  const speedScore = meanOfFinite([imaginedSpeedScore, observedE2eSpeedScore]);
  if (
    intelligenceScore == null &&
    agenticScore == null &&
    priceScore == null &&
    speedScore == null
  ) {
    return null;
  }
  return {
    intelligence_score: intelligenceScore,
    agentic_score: agenticScore,
    speed_score: speedScore,
    price_score: priceScore,
  };
}

export function attachPercentiles(
  models: ModelStatsSelectedModel[],
): ModelStatsSelectedModel[] {
  const intelligenceScores = models.map((model) =>
    asFiniteNumber(asRecord(model.scores).intelligence_score),
  );
  const agenticScores = models.map((model) =>
    asFiniteNumber(asRecord(model.scores).agentic_score),
  );
  const speedScores = models.map((model) =>
    asFiniteNumber(asRecord(model.scores).speed_score),
  );
  const priceScores = models.map((model) =>
    asFiniteNumber(asRecord(model.scores).price_score),
  );

  return models.map((model) => {
    const scores = asRecord(model.scores);
    const intelligenceScore = asFiniteNumber(scores.intelligence_score);
    const agenticScore = asFiniteNumber(scores.agentic_score);
    const speedScore = asFiniteNumber(scores.speed_score);
    const priceScore = asFiniteNumber(scores.price_score);
    const intelligencePercentile =
      intelligenceScore == null
        ? null
        : percentileRank(intelligenceScores, intelligenceScore);
    const agenticPercentile =
      agenticScore == null ? null : percentileRank(agenticScores, agenticScore);
    const speedPercentile =
      speedScore == null ? null : percentileRank(speedScores, speedScore);
    const pricePercentile =
      priceScore == null ? null : percentileRank(priceScores, priceScore);
    return {
      ...model,
      percentiles: {
        intelligence_percentile: intelligencePercentile,
        agentic_percentile: agenticPercentile,
        speed_percentile: speedPercentile,
        price_percentile: pricePercentile,
      },
    };
  });
}
