import { normalizeModelToken } from "../shared.js";

import { type LlmMatchCandidate } from "./types.js";
import {
  commonPrefixLength,
  firstParsedNumber,
  isNumericToken,
  parseActiveBToken,
  parseBScaleToken,
  parseNumericOrBScaleToken,
  splitBaseModelId,
  splitTokens,
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
  artificialAnalysisSlug: string,
  modelsDevModelId: string,
): number {
  const artificialAnalysisTokens = splitTokens(artificialAnalysisSlug);
  const modelTokens = splitTokens(splitBaseModelId(modelsDevModelId));
  const maxLength = Math.min(
    artificialAnalysisTokens.length,
    modelTokens.length,
  );
  for (let tokenIndex = 0; tokenIndex < maxLength; tokenIndex += 1) {
    const artificialAnalysisToken = artificialAnalysisTokens[tokenIndex];
    const modelToken = modelTokens[tokenIndex];
    const artificialAnalysisNumericValue = parseNumericOrBScaleToken(
      artificialAnalysisToken,
    );
    const modelNumericValue = parseNumericOrBScaleToken(modelToken);
    if (artificialAnalysisNumericValue != null && modelNumericValue != null) {
      return artificialAnalysisNumericValue === modelNumericValue
        ? NUMERIC_EXACT_MATCH_REWARD
        : 0;
    }
  }
  return 0;
}

function numericClosenessReward(
  artificialAnalysisSlug: string,
  modelsDevModelId: string,
): number {
  const artificialAnalysisNumbers = splitTokens(artificialAnalysisSlug)
    .map((token) => parseNumericOrBScaleToken(token))
    .filter((value): value is number => value != null);
  const modelNumbers = splitTokens(splitBaseModelId(modelsDevModelId))
    .map((token) => parseNumericOrBScaleToken(token))
    .filter((value): value is number => value != null);

  const maxLength = Math.max(
    artificialAnalysisNumbers.length,
    modelNumbers.length,
  );
  for (let numberIndex = 0; numberIndex < maxLength; numberIndex += 1) {
    const artificialAnalysisValue = artificialAnalysisNumbers[numberIndex];
    const modelValue = modelNumbers[numberIndex];
    if (artificialAnalysisValue == null || modelValue == null) {
      return 0;
    }
    if (artificialAnalysisValue === modelValue) {
      continue;
    }
    return (
      NUMERIC_CLOSENESS_REWARD_SCALE /
      (1 + Math.abs(artificialAnalysisValue - modelValue))
    );
  }
  return NUMERIC_ALL_EQUAL_REWARD;
}

function bScaleRewardOrPenalty(
  artificialAnalysisSlug: string,
  modelsDevModelId: string,
  modelsDevModelName: string,
): number {
  const artificialAnalysisTokens = splitTokens(artificialAnalysisSlug);
  const modelBaseTokens = splitTokens(splitBaseModelId(modelsDevModelId));
  const modelNameTokens = splitTokens(modelsDevModelName);

  const artificialAnalysisBScale = firstParsedNumber(
    artificialAnalysisTokens,
    parseBScaleToken,
  );
  if (artificialAnalysisBScale == null) {
    return 0;
  }

  const baseBScale = firstParsedNumber(modelBaseTokens, parseBScaleToken);
  const nameBScale = firstParsedNumber(modelNameTokens, parseBScaleToken);
  const candidateBScale = baseBScale ?? nameBScale;
  if (candidateBScale == null) {
    return -B_SCALE_MISSING_PENALTY;
  }
  if (candidateBScale === artificialAnalysisBScale) {
    return B_SCALE_EXACT_REWARD;
  }
  return -B_SCALE_MISMATCH_PENALTY;
}

function hasHardBScaleMismatch(
  artificialAnalysisSlug: string,
  modelsDevModelId: string,
  modelsDevModelName: string,
): boolean {
  const artificialAnalysisBScale = firstParsedNumber(
    splitTokens(artificialAnalysisSlug),
    parseBScaleToken,
  );
  if (artificialAnalysisBScale == null) {
    return false;
  }

  const modelBaseBScale = firstParsedNumber(
    splitTokens(splitBaseModelId(modelsDevModelId)),
    parseBScaleToken,
  );
  const modelNameBScale = firstParsedNumber(
    splitTokens(modelsDevModelName),
    parseBScaleToken,
  );
  const candidateBScale = modelBaseBScale ?? modelNameBScale;
  if (candidateBScale == null) {
    return false;
  }

  return candidateBScale !== artificialAnalysisBScale;
}

function activeBRewardOrPenalty(
  artificialAnalysisSlug: string,
  modelsDevModelId: string,
  modelsDevModelName: string,
): number {
  const artificialAnalysisTokens = splitTokens(artificialAnalysisSlug);
  const modelBaseTokens = splitTokens(splitBaseModelId(modelsDevModelId));
  const modelNameTokens = splitTokens(modelsDevModelName);

  const artificialAnalysisActiveB = firstParsedNumber(
    artificialAnalysisTokens,
    parseActiveBToken,
  );
  if (artificialAnalysisActiveB == null) {
    return 0;
  }

  const baseActiveB = firstParsedNumber(modelBaseTokens, parseActiveBToken);
  const nameActiveB = firstParsedNumber(modelNameTokens, parseActiveBToken);
  const candidateActiveB = baseActiveB ?? nameActiveB;
  if (candidateActiveB == null) {
    return 0;
  }
  if (candidateActiveB === artificialAnalysisActiveB) {
    return ACTIVE_B_EXACT_REWARD;
  }
  return -ACTIVE_B_MISMATCH_PENALTY;
}

function sameVariantReward(
  artificialAnalysisSlug: string,
  modelsDevModelId: string,
  modelsDevModelName: string,
): number {
  const artificialAnalysisTokens = splitTokens(artificialAnalysisSlug);
  const modelBaseTokens = splitTokens(splitBaseModelId(modelsDevModelId));
  const modelNameTokens = splitTokens(modelsDevModelName);
  const artificialAnalysisLastToken = artificialAnalysisTokens.at(-1);
  if (
    !artificialAnalysisLastToken ||
    isNumericToken(artificialAnalysisLastToken)
  ) {
    return 0;
  }
  const baseLastToken = modelBaseTokens.at(-1);
  const nameLastToken = modelNameTokens.at(-1);
  if (
    artificialAnalysisLastToken === baseLastToken ||
    artificialAnalysisLastToken === nameLastToken
  ) {
    return VARIANT_SUFFIX_REWARD;
  }
  return 0;
}

function coverageRewardOrPenalty(
  artificialAnalysisSlug: string,
  modelsDevModelId: string,
  modelsDevModelName: string,
): number {
  const artificialAnalysisSet = new Set(splitTokens(artificialAnalysisSlug));
  const baseSet = new Set(splitTokens(splitBaseModelId(modelsDevModelId)));
  const nameSet = new Set(splitTokens(modelsDevModelName));

  function compareSets(candidateSet: Set<string>): number {
    if (artificialAnalysisSet.size === 0) {
      return 0;
    }
    const missingCount = [...artificialAnalysisSet].filter(
      (token) => !candidateSet.has(token),
    ).length;
    if (missingCount > 0) {
      return -COVERAGE_MISSING_BASE_PENALTY - missingCount;
    }
    if (candidateSet.size === artificialAnalysisSet.size) {
      return COVERAGE_EXACT_REWARD;
    }
    return 0;
  }

  return Math.max(compareSets(baseSet), compareSets(nameSet));
}

export function hasFirstTokenMatch(
  artificialAnalysisSlug: string,
  modelsDevModelId: string,
  modelsDevModelName: string,
): boolean {
  // Guardrail: first-token mismatch usually means wrong model family.
  const artificialAnalysisFirstToken = splitTokens(artificialAnalysisSlug)[0];
  if (!artificialAnalysisFirstToken) {
    return false;
  }
  return (
    artificialAnalysisFirstToken ===
      splitTokens(splitBaseModelId(modelsDevModelId))[0] ||
    artificialAnalysisFirstToken === splitTokens(modelsDevModelName)[0]
  );
}

export function scoreCandidate(
  artificialAnalysisSlug: string,
  modelsDevModelId: string,
  modelsDevModelName: string,
): number {
  // Prefix reward addresses cross-family false positives.
  const normalizedArtificialAnalysisSlug = normalizeModelToken(
    artificialAnalysisSlug,
  );
  const normalizedModelBase = normalizeModelToken(
    splitBaseModelId(modelsDevModelId),
  );
  const normalizedModelName = normalizeModelToken(modelsDevModelName);
  const artificialAnalysisTokens = splitTokens(artificialAnalysisSlug);
  const modelBaseTokens = splitTokens(splitBaseModelId(modelsDevModelId));
  const modelNameTokens = splitTokens(modelsDevModelName);
  const basePrefixLength = commonPrefixLength(
    normalizedArtificialAnalysisSlug,
    normalizedModelBase,
  );
  const modelNamePrefixLength = commonPrefixLength(
    normalizedArtificialAnalysisSlug,
    normalizedModelName,
  );
  const maxPrefixLength = Math.max(basePrefixLength, modelNamePrefixLength);
  if (maxPrefixLength === 0) {
    return 0;
  }
  if (
    hasHardBScaleMismatch(
      artificialAnalysisSlug,
      modelsDevModelId,
      modelsDevModelName,
    )
  ) {
    return 0;
  }

  const weightedTokenScore = Math.max(
    weightedTokenPrefixScore(artificialAnalysisTokens, modelBaseTokens),
    weightedTokenPrefixScore(artificialAnalysisTokens, modelNameTokens),
  );

  // Numeric reward keeps nearby versions ordered (e.g. 5.2 > 5.1 when 5.3 is missing).
  // Variant reward keeps suffix-sensitive families aligned (codex/haiku/opus).
  // Coverage penalty suppresses unrelated but superficially similar names.
  return (
    weightedTokenScore * TOKEN_PREFIX_REWARD_MULTIPLIER +
    numericMatchReward(artificialAnalysisSlug, modelsDevModelId) +
    numericClosenessReward(artificialAnalysisSlug, modelsDevModelId) +
    sameVariantReward(
      artificialAnalysisSlug,
      modelsDevModelId,
      modelsDevModelName,
    ) +
    bScaleRewardOrPenalty(
      artificialAnalysisSlug,
      modelsDevModelId,
      modelsDevModelName,
    ) +
    activeBRewardOrPenalty(
      artificialAnalysisSlug,
      modelsDevModelId,
      modelsDevModelName,
    ) +
    coverageRewardOrPenalty(
      artificialAnalysisSlug,
      modelsDevModelId,
      modelsDevModelName,
    ) +
    maxPrefixLength * CHAR_PREFIX_REWARD_SCALE -
    Math.abs(
      normalizedArtificialAnalysisSlug.length - normalizedModelBase.length,
    ) *
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
