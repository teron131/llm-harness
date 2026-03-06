import {
  FALLBACK_PROVIDER_IDS,
  PRIMARY_PROVIDER_ID,
  asRecord,
  modelSlugFromModelId,
  normalizeModelToken,
} from "../shared.js";

import {
  compareCandidates,
  hasFirstTokenMatch,
  scoreCandidate,
} from "./scoring.js";
import { splitBaseModelId } from "./tokenize.js";
import {
  type ArtificialAnalysisModel,
  type LlmMatchCandidate,
  type LlmMatchResult,
  type MatcherInputModel,
  type MatcherRunOutput,
  type ModelsDevModel,
  type PreferredProviderScopedModels,
} from "./types.js";

const VOID_THRESHOLD_RANGE_RATIO = 0.35;

export function uniqueModelCount(modelsDevModels: ModelsDevModel[]): number {
  return new Set(
    modelsDevModels.map((modelsDevModel) => modelsDevModel.model_id),
  ).size;
}

function hasExactSlugFallbackCandidate(
  artificialAnalysisSlug: string,
  fallbackCandidates: LlmMatchCandidate[],
): boolean {
  const normalizedArtificialAnalysisSlug = normalizeModelToken(
    artificialAnalysisSlug,
  );
  if (normalizedArtificialAnalysisSlug.length === 0) {
    return false;
  }
  return fallbackCandidates.some((candidate) => {
    const candidateSlug = normalizeModelToken(
      splitBaseModelId(candidate.model_id),
    );
    return (
      candidateSlug.length > 0 &&
      candidateSlug === normalizedArtificialAnalysisSlug
    );
  });
}

export function splitPreferredProviderModels(
  modelsDevModels: ModelsDevModel[],
): PreferredProviderScopedModels {
  const primary = modelsDevModels.filter(
    (modelsDevModel) => modelsDevModel.provider_id === PRIMARY_PROVIDER_ID,
  );
  const fallback = modelsDevModels.filter((modelsDevModel) =>
    FALLBACK_PROVIDER_IDS.has(modelsDevModel.provider_id),
  );
  return { primary, fallback };
}

function collectCandidatesForArtificialAnalysisSlug(
  artificialAnalysisSlug: string,
  modelsDevModels: ModelsDevModel[],
): LlmMatchCandidate[] {
  if (!artificialAnalysisSlug) {
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
          artificialAnalysisSlug,
          modelsDevModel.model_id,
          modelsDevModelName,
        )
      ) {
        return null;
      }
      const candidateScore = scoreCandidate(
        artificialAnalysisSlug,
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
  artificialAnalysisSlug: string,
  scopedModels: PreferredProviderScopedModels,
): LlmMatchCandidate[] {
  const primaryCandidates = collectCandidatesForArtificialAnalysisSlug(
    artificialAnalysisSlug,
    scopedModels.primary,
  );
  const fallbackCandidates = collectCandidatesForArtificialAnalysisSlug(
    artificialAnalysisSlug,
    scopedModels.fallback,
  );
  if (primaryCandidates.length === 0) {
    return fallbackCandidates;
  }
  if (
    hasExactSlugFallbackCandidate(artificialAnalysisSlug, fallbackCandidates)
  ) {
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

export function runMatcher(
  sourceModels: MatcherInputModel[],
  scopedModels: PreferredProviderScopedModels,
  maxCandidates: number,
): MatcherRunOutput {
  const models = sourceModels.map((sourceModel) => {
    const candidates = selectPreferredCandidatesForArtificialAnalysisSlug(
      sourceModel.artificialAnalysisSlug,
      scopedModels,
    ).slice(0, maxCandidates);
    return {
      artificial_analysis_slug: sourceModel.artificialAnalysisSlug,
      artificial_analysis_name: sourceModel.artificialAnalysisName,
      artificial_analysis_release_date:
        sourceModel.artificialAnalysisReleaseDate,
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

export function buildInputModelsFromArtificialAnalysis(
  artificialAnalysisModels: ArtificialAnalysisModel[],
): MatcherInputModel[] {
  return artificialAnalysisModels.map((artificialAnalysisModel) => ({
    artificialAnalysisSlug:
      typeof artificialAnalysisModel.slug === "string"
        ? artificialAnalysisModel.slug
        : "",
    artificialAnalysisName:
      typeof artificialAnalysisModel.name === "string"
        ? artificialAnalysisModel.name
        : null,
    artificialAnalysisReleaseDate:
      typeof artificialAnalysisModel.release_date === "string"
        ? artificialAnalysisModel.release_date
        : null,
  }));
}

export function buildInputModelsFromScrapedRows(
  scrapedRows: unknown[],
): MatcherInputModel[] {
  return scrapedRows.map((scrapedRow) => {
    const scrapedRowRecord = asRecord(scrapedRow);
    const modelId =
      typeof scrapedRowRecord.model_id === "string"
        ? scrapedRowRecord.model_id
        : null;
    const artificialAnalysisSlug = modelSlugFromModelId(modelId) ?? "";
    return {
      artificialAnalysisSlug,
      artificialAnalysisName: modelId,
      artificialAnalysisReleaseDate: null,
    };
  });
}
