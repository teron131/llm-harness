/** Score helpers for LLM stats: keep raw score formulas here and attach normalized relative scores later. */
import { percentileRank } from "../../utils.js";
import { asFiniteNumber, asRecord, type JsonObject } from "../shared.js";

import { type ModelStatsSelectedModel, type ScoringConfig } from "./types.js";

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

/** Estimate a blended price from raw pricing fields, preferring weighted OpenRouter pricing when available. */
export function blendedPriceValue(
  costLike: unknown,
  scoringConfig: ScoringConfig,
): number | null {
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
      scoringConfig.weightedPriceInputRatio * effectiveInputCost +
      scoringConfig.weightedPriceOutputRatio * effectiveOutputCost
    );
  }

  const cacheWeightedInput = cacheReadCost != null ? cacheReadCost : inputCost;
  const cacheWeightedOutput =
    cacheWriteCost != null
      ? 0.1 * cacheWriteCost + 0.9 * outputCost
      : outputCost;
  const baseProxy =
    scoringConfig.weightedPriceInputRatio *
      (scoringConfig.weightedPriceInputRatio * cacheWeightedInput +
        scoringConfig.weightedPriceOutputRatio * inputCost) +
    scoringConfig.weightedPriceOutputRatio * cacheWeightedOutput;

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
    scoringConfig.weightedPriceInputRatio *
      (scoringConfig.weightedPriceInputRatio * over200kInputWeighted +
        scoringConfig.weightedPriceOutputRatio * over200kInput) +
    scoringConfig.weightedPriceOutputRatio * over200kOutputWeighted;

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

/** Derive representative output-token anchors from OpenRouter latency/throughput observations. */
export function deriveSpeedOutputTokenAnchors(
  openRouterSpeedById: Map<string, JsonObject>,
  scoringConfig: ScoringConfig,
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
    return [...scoringConfig.defaultSpeedOutputTokenAnchors];
  }

  const q0 = impliedTokenUsages[0] ?? null;
  const [q1, q2, q3] = scoringConfig.speedAnchorQuantiles.map((quantile) =>
    quantileFromSorted(impliedTokenUsages, quantile),
  );
  const q4 = impliedTokenUsages[impliedTokenUsages.length - 1] ?? null;
  const numericQuantileAnchors = [q0, q1, q2, q3, q4].filter(
    (value): value is number => value != null && Number.isFinite(value),
  );
  if (numericQuantileAnchors.length !== 5) {
    return [...scoringConfig.defaultSpeedOutputTokenAnchors];
  }

  const sourceMin = numericQuantileAnchors[0] as number;
  const sourceMax = numericQuantileAnchors[
    numericQuantileAnchors.length - 1
  ] as number;
  if (!(sourceMax > sourceMin)) {
    return [...scoringConfig.defaultSpeedOutputTokenAnchors];
  }

  return numericQuantileAnchors.map((anchor) => {
    const normalized = (anchor - sourceMin) / (sourceMax - sourceMin);
    const mapped =
      scoringConfig.speedOutputTokenRangeMin +
      normalized *
        (scoringConfig.speedOutputTokenRangeMax -
          scoringConfig.speedOutputTokenRangeMin);
    return Math.round(mapped);
  });
}

/** Compute the raw score bundle for a single final model row. */
export function buildScores(
  model: JsonObject,
  cost: unknown,
  speed: JsonObject,
  speedOutputTokenAnchors: number[],
  scoringConfig: ScoringConfig,
): unknown {
  const intelligenceIndex =
    metricValue(model, "intelligence_index") ??
    metricValue(model, "artificial_analysis_intelligence_index");
  const agenticIndex =
    metricValue(model, "agentic_index") ??
    metricValue(model, "artificial_analysis_agentic_index");
  const intelligenceBenchmarkMean = meanOfFinite(
    scoringConfig.intelligenceBenchmarkKeys.map((key) =>
      metricValue(model, key),
    ),
  );
  const agenticBenchmarkMean = meanOfFinite(
    scoringConfig.agenticBenchmarkKeys.map((key) => metricValue(model, key)),
  );
  const intelligenceScore =
    intelligenceIndex != null && intelligenceBenchmarkMean != null
      ? (intelligenceIndex + intelligenceBenchmarkMean * 100) / 2
      : null;
  const agenticScore =
    agenticIndex != null && agenticBenchmarkMean != null
      ? (agenticIndex + agenticBenchmarkMean * 100) / 2
      : null;
  const blendedPrice = blendedPriceValue(cost, scoringConfig);
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

function minMaxScale(
  values: Array<number | null>,
  value: number | null,
): number | null {
  if (value == null) {
    return null;
  }
  const finiteValues = values.filter(
    (candidate): candidate is number =>
      candidate != null && Number.isFinite(candidate),
  );
  if (finiteValues.length === 0) {
    return null;
  }
  const minValue = Math.min(...finiteValues);
  const maxValue = Math.max(...finiteValues);
  if (maxValue === minValue) {
    return 100;
  }
  return ((value - minValue) / (maxValue - minValue)) * 100;
}

/** Attach normalized `relative_scores` using min-max for intelligence/agentic and percentiles for speed/price. */
export function attachRelativeScores(
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
    const intelligenceRelativeScore = minMaxScale(
      intelligenceScores,
      intelligenceScore,
    );
    const agenticRelativeScore = minMaxScale(agenticScores, agenticScore);
    const speedRelativeScore =
      speedScore == null ? null : percentileRank(speedScores, speedScore);
    const priceRelativeScore =
      priceScore == null ? null : percentileRank(priceScores, priceScore);
    const overallRelativeScore = meanOfFinite([
      intelligenceRelativeScore,
      agenticRelativeScore,
      speedRelativeScore,
      priceRelativeScore,
    ]);
    return {
      ...model,
      relative_scores: {
        intelligence_score: intelligenceRelativeScore,
        agentic_score: agenticRelativeScore,
        speed_score: speedRelativeScore,
        price_score: priceRelativeScore,
        overall_score: overallRelativeScore,
      },
    };
  });
}
