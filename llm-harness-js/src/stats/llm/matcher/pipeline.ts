/** Internal matcher pipeline: scope provider pools, collect candidates, and apply the final void threshold. */
import {
  FALLBACK_PROVIDER_IDS,
  PRIMARY_PROVIDER_ID,
  normalizeModelToken,
} from "../shared.js";

import {
  compareCandidates,
  hasFirstTokenMatch,
  scoreCandidate,
} from "./scoring.js";
import { splitBaseModelId } from "./tokenize.js";
import {
  type LlmMatchCandidate,
  type LlmMatchResult,
  type MatcherSourceModel,
  type MatcherRunOutput,
  type ModelsDevModel,
  type PreferredProviderPools,
} from "./types.js";

const VOID_THRESHOLD_RANGE_RATIO = 0.35;

/** Count unique models.dev model ids after provider scoping. */
export function uniqueModelCount(modelsDevModels: ModelsDevModel[]): number {
  return new Set(
    modelsDevModels.map((modelsDevModel) => modelsDevModel.model_id),
  ).size;
}

function hasExactSlugFallbackCandidate(
  sourceSlug: string,
  fallbackCandidates: LlmMatchCandidate[],
): boolean {
  const normalizedSourceSlug = normalizeModelToken(sourceSlug);
  if (normalizedSourceSlug.length === 0) {
    return false;
  }
  return fallbackCandidates.some((candidate) => {
    const candidateSlug = normalizeModelToken(
      splitBaseModelId(candidate.model_id),
    );
    return candidateSlug.length > 0 && candidateSlug === normalizedSourceSlug;
  });
}

/** Split models.dev rows into primary and fallback provider pools for deterministic matcher behavior. */
export function splitPreferredProviderModels(
  modelsDevModels: ModelsDevModel[],
): PreferredProviderPools {
  const primary = modelsDevModels.filter(
    (modelsDevModel) => modelsDevModel.provider_id === PRIMARY_PROVIDER_ID,
  );
  const fallback = modelsDevModels.filter((modelsDevModel) =>
    FALLBACK_PROVIDER_IDS.has(modelsDevModel.provider_id),
  );
  return { primary, fallback };
}

function collectCandidatesForSourceSlug(
  sourceSlug: string,
  modelsDevModels: ModelsDevModel[],
): LlmMatchCandidate[] {
  if (!sourceSlug) {
    return [];
  }

  return modelsDevModels
    .map((modelsDevModel) => {
      const modelsDevModelName =
        typeof modelsDevModel.model.name === "string"
          ? modelsDevModel.model.name
          : "";
      if (
        !hasFirstTokenMatch(
          sourceSlug,
          modelsDevModel.model_id,
          modelsDevModelName,
        )
      ) {
        return null;
      }
      const candidateScore = scoreCandidate(
        sourceSlug,
        modelsDevModel.model_id,
        modelsDevModelName,
      );
      if (candidateScore <= 0) {
        return null;
      }
      return {
        model_id: modelsDevModel.model_id,
        provider_id: modelsDevModel.provider_id,
        provider_name: modelsDevModel.provider_name,
        model_name: modelsDevModelName || null,
        score: candidateScore,
      };
    })
    .filter((candidate): candidate is LlmMatchCandidate => candidate != null)
    .sort(compareCandidates);
}

function selectPreferredCandidatesForArtificialAnalysisSlug(
  sourceSlug: string,
  providerPools: PreferredProviderPools,
): LlmMatchCandidate[] {
  const primaryCandidates = collectCandidatesForSourceSlug(
    sourceSlug,
    providerPools.primary,
  );
  const fallbackCandidates = collectCandidatesForSourceSlug(
    sourceSlug,
    providerPools.fallback,
  );
  if (primaryCandidates.length === 0) {
    return fallbackCandidates;
  }
  if (hasExactSlugFallbackCandidate(sourceSlug, fallbackCandidates)) {
    return fallbackCandidates;
  }
  return primaryCandidates;
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

/** Run the matcher over source rows and return ranked candidates plus post-void summary counts. */
export function runMatcher(
  sourceModels: MatcherSourceModel[],
  providerPools: PreferredProviderPools,
  maxCandidates: number,
): MatcherRunOutput {
  const models = sourceModels.map((sourceModel) => {
    const candidates = selectPreferredCandidatesForArtificialAnalysisSlug(
      sourceModel.sourceSlug,
      providerPools,
    ).slice(0, maxCandidates);
    return {
      artificial_analysis_slug: sourceModel.sourceSlug,
      artificial_analysis_name: sourceModel.sourceName,
      artificial_analysis_release_date: sourceModel.sourceReleaseDate,
      best_match: candidates[0] ?? null,
      candidates,
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
    models,
    voidThreshold: voidStats.threshold,
    voidedCount: voidStats.voided,
    preVoidMatchedCount,
    preVoidUnmatchedCount,
    matchedCount,
    unmatchedCount,
  };
}
