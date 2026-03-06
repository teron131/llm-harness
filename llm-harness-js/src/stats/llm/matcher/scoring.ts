import { normalizeModelToken } from "../shared.js";

import { type LlmMatchCandidate } from "./types.js";
import {
  commonPrefixLength,
  firstParsedNumber,
  isNumericToken,
  parseActiveBToken,
  parseBScaleToken,
  splitBaseModelId,
  splitBaseModelTokens,
  splitTokens,
  parsedNumericTokens,
} from "./tokenize.js";

const TOKEN_PREFIX_WEIGHTS = [5, 4, 3, 2, 1] as const;
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

function weightedTokenPrefixScore(
  leftTokens: string[],
  rightTokens: string[],
): number {
  const maxLength = Math.min(leftTokens.length, rightTokens.length);
  let score = 0;
  for (let tokenIndex = 0; tokenIndex < maxLength; tokenIndex += 1) {
    if (leftTokens[tokenIndex] !== rightTokens[tokenIndex]) {
      break;
    }
    score += TOKEN_PREFIX_WEIGHTS[tokenIndex] ?? 0;
  }
  return score;
}

function numericMatchReward(
  sourceSlug: string,
  candidateModelId: string,
): number {
  const sourceTokens = splitTokens(sourceSlug);
  const candidateTokens = splitBaseModelTokens(candidateModelId);
  const maxLength = Math.min(sourceTokens.length, candidateTokens.length);
  for (let tokenIndex = 0; tokenIndex < maxLength; tokenIndex += 1) {
    const sourceValue = parsedNumericTokens([
      sourceTokens[tokenIndex] ?? "",
    ])[0];
    const candidateValue = parsedNumericTokens([
      candidateTokens[tokenIndex] ?? "",
    ])[0];
    if (sourceValue != null && candidateValue != null) {
      return sourceValue === candidateValue ? NUMERIC_EXACT_MATCH_REWARD : 0;
    }
  }
  return 0;
}

function numericClosenessReward(
  sourceSlug: string,
  candidateModelId: string,
): number {
  const sourceNumbers = parsedNumericTokens(splitTokens(sourceSlug));
  const candidateNumbers = parsedNumericTokens(
    splitBaseModelTokens(candidateModelId),
  );

  const maxLength = Math.max(sourceNumbers.length, candidateNumbers.length);
  for (let numberIndex = 0; numberIndex < maxLength; numberIndex += 1) {
    const sourceValue = sourceNumbers[numberIndex];
    const candidateValue = candidateNumbers[numberIndex];
    if (sourceValue == null || candidateValue == null) {
      return 0;
    }
    if (sourceValue === candidateValue) {
      continue;
    }
    return (
      NUMERIC_CLOSENESS_REWARD_SCALE /
      (1 + Math.abs(sourceValue - candidateValue))
    );
  }
  return NUMERIC_ALL_EQUAL_REWARD;
}

function candidateScaleValue(
  candidateModelId: string,
  candidateModelName: string,
  parser: (token: string | undefined) => number | null,
): number | null {
  const baseValue = firstParsedNumber(
    splitBaseModelTokens(candidateModelId),
    parser,
  );
  const nameValue = firstParsedNumber(splitTokens(candidateModelName), parser);
  return baseValue ?? nameValue;
}

function bScaleRewardOrPenalty(
  sourceSlug: string,
  candidateModelId: string,
  candidateModelName: string,
): number {
  const sourceBScale = firstParsedNumber(
    splitTokens(sourceSlug),
    parseBScaleToken,
  );
  if (sourceBScale == null) {
    return 0;
  }
  const candidateBScale = candidateScaleValue(
    candidateModelId,
    candidateModelName,
    parseBScaleToken,
  );
  if (candidateBScale == null) {
    return -B_SCALE_MISSING_PENALTY;
  }
  if (candidateBScale === sourceBScale) {
    return B_SCALE_EXACT_REWARD;
  }
  return -B_SCALE_MISMATCH_PENALTY;
}

function hasHardBScaleMismatch(
  sourceSlug: string,
  candidateModelId: string,
  candidateModelName: string,
): boolean {
  const sourceBScale = firstParsedNumber(
    splitTokens(sourceSlug),
    parseBScaleToken,
  );
  if (sourceBScale == null) {
    return false;
  }
  const candidateBScale = candidateScaleValue(
    candidateModelId,
    candidateModelName,
    parseBScaleToken,
  );
  if (candidateBScale == null) {
    return false;
  }
  return candidateBScale !== sourceBScale;
}

function activeBRewardOrPenalty(
  sourceSlug: string,
  candidateModelId: string,
  candidateModelName: string,
): number {
  const sourceActiveB = firstParsedNumber(
    splitTokens(sourceSlug),
    parseActiveBToken,
  );
  if (sourceActiveB == null) {
    return 0;
  }
  const candidateActiveB = candidateScaleValue(
    candidateModelId,
    candidateModelName,
    parseActiveBToken,
  );
  if (candidateActiveB == null) {
    return 0;
  }
  if (candidateActiveB === sourceActiveB) {
    return ACTIVE_B_EXACT_REWARD;
  }
  return -ACTIVE_B_MISMATCH_PENALTY;
}

function sameVariantReward(
  sourceSlug: string,
  candidateModelId: string,
  candidateModelName: string,
): number {
  const sourceLastToken = splitTokens(sourceSlug).at(-1);
  if (!sourceLastToken || isNumericToken(sourceLastToken)) {
    return 0;
  }
  const baseLastToken = splitBaseModelTokens(candidateModelId).at(-1);
  const nameLastToken = splitTokens(candidateModelName).at(-1);
  if (sourceLastToken === baseLastToken || sourceLastToken === nameLastToken) {
    return VARIANT_SUFFIX_REWARD;
  }
  return 0;
}

function coverageRewardOrPenalty(
  sourceSlug: string,
  candidateModelId: string,
  candidateModelName: string,
): number {
  const sourceSet = new Set(splitTokens(sourceSlug));
  const baseSet = new Set(splitBaseModelTokens(candidateModelId));
  const nameSet = new Set(splitTokens(candidateModelName));

  function compareSets(candidateSet: Set<string>): number {
    if (sourceSet.size === 0) {
      return 0;
    }
    const missingCount = [...sourceSet].filter(
      (token) => !candidateSet.has(token),
    ).length;
    if (missingCount > 0) {
      return -COVERAGE_MISSING_BASE_PENALTY - missingCount;
    }
    if (candidateSet.size === sourceSet.size) {
      return COVERAGE_EXACT_REWARD;
    }
    return 0;
  }

  return Math.max(compareSets(baseSet), compareSets(nameSet));
}

export function hasFirstTokenMatch(
  sourceSlug: string,
  candidateModelId: string,
  candidateModelName: string,
): boolean {
  // Guardrail: first-token mismatch usually means wrong model family.
  const sourceFirstToken = splitTokens(sourceSlug)[0];
  if (!sourceFirstToken) {
    return false;
  }
  return (
    sourceFirstToken === splitBaseModelTokens(candidateModelId)[0] ||
    sourceFirstToken === splitTokens(candidateModelName)[0]
  );
}

export function scoreCandidate(
  sourceSlug: string,
  candidateModelId: string,
  candidateModelName: string,
): number {
  // Prefix reward addresses cross-family false positives.
  const normalizedSourceSlug = normalizeModelToken(sourceSlug);
  const normalizedModelBase = normalizeModelToken(
    splitBaseModelId(candidateModelId),
  );
  const normalizedModelName = normalizeModelToken(candidateModelName);
  const sourceTokens = splitTokens(sourceSlug);
  const modelBaseTokens = splitBaseModelTokens(candidateModelId);
  const modelNameTokens = splitTokens(candidateModelName);
  const basePrefixLength = commonPrefixLength(
    normalizedSourceSlug,
    normalizedModelBase,
  );
  const modelNamePrefixLength = commonPrefixLength(
    normalizedSourceSlug,
    normalizedModelName,
  );
  const maxPrefixLength = Math.max(basePrefixLength, modelNamePrefixLength);
  if (maxPrefixLength === 0) {
    return 0;
  }
  if (hasHardBScaleMismatch(sourceSlug, candidateModelId, candidateModelName)) {
    return 0;
  }

  const weightedTokenScore = Math.max(
    weightedTokenPrefixScore(sourceTokens, modelBaseTokens),
    weightedTokenPrefixScore(sourceTokens, modelNameTokens),
  );

  // Numeric reward keeps nearby versions ordered (e.g. 5.2 > 5.1 when 5.3 is missing).
  // Variant reward keeps suffix-sensitive families aligned (codex/haiku/opus).
  // Coverage penalty suppresses unrelated but superficially similar names.
  return (
    weightedTokenScore * TOKEN_PREFIX_REWARD_MULTIPLIER +
    numericMatchReward(sourceSlug, candidateModelId) +
    numericClosenessReward(sourceSlug, candidateModelId) +
    sameVariantReward(sourceSlug, candidateModelId, candidateModelName) +
    bScaleRewardOrPenalty(sourceSlug, candidateModelId, candidateModelName) +
    activeBRewardOrPenalty(sourceSlug, candidateModelId, candidateModelName) +
    coverageRewardOrPenalty(sourceSlug, candidateModelId, candidateModelName) +
    maxPrefixLength * CHAR_PREFIX_REWARD_SCALE -
    Math.abs(normalizedSourceSlug.length - normalizedModelBase.length) *
      LENGTH_GAP_PENALTY_SCALE
  );
}

export function compareCandidates(
  left: LlmMatchCandidate,
  right: LlmMatchCandidate,
): number {
  if (left.score !== right.score) {
    return right.score - left.score;
  }
  return left.model_id.localeCompare(right.model_id);
}
