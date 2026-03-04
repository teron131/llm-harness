import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { OpenRouterEmbeddings } from "../../../clients/openrouter";
import { getArtificialAnalysisImageStats } from "./artificial-analysis";
import { getArenaAiImageStats } from "./arena-ai";

type ArtificialAnalysisImageModel = Awaited<
  ReturnType<typeof getArtificialAnalysisImageStats>
>["data"][number];
type ArenaAiImageModel = Awaited<
  ReturnType<typeof getArenaAiImageStats>
>["rows"][number];

const DEFAULT_MAX_CANDIDATES = 3;
const PROVIDER_MATCH_REWARD = 2;
const MIN_ACCEPTED_CANDIDATE_SCORE = 3;
const VOID_THRESHOLD_RANGE_RATIO = 0.12;
const TOKEN_COVERAGE_WEIGHT = 8;
const QUALIFIER_MATCH_WEIGHT = 2.5;
const QUALIFIER_MISS_PENALTY = 2;
const MAX_QUALIFIER_PENALTY = 6;
const RANK_PROXIMITY_RADIUS = 10;
const RANK_PROXIMITY_MAX_BONUS = 3;
const TOP_RANK_PROTECTION_COUNT = 20;
const TOP_RANK_PROTECTION_MARGIN = 0.6;
const TOP_RANK_PROTECTION_THRESHOLD_DELTA = 0.6;
const DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small";
const DEFAULT_EMBEDDING_WEIGHT = 4;
const EMBEDDING_BATCH_SIZE = 64;
const EMBEDDING_CACHE_PATH = resolve(".cache/image_match_embedding_cache.json");

const NOISE_TOKENS = new Set([
  "image",
  "images",
  "model",
  "models",
  "generate",
  "generation",
  "preview",
  "version",
  "ver",
  "ai",
  "the",
  "and",
  "for",
  "with",
]);

const QUALIFIER_TOKENS = new Set([
  "ultra",
  "max",
  "pro",
  "mini",
  "dev",
  "fast",
  "flash",
  "standard",
  "flex",
  "turbo",
  "lite",
  "instruct",
  "high",
  "low",
  "medium",
  "plus",
  "base",
]);

const PROVIDER_NOISE_TOKENS = new Set([
  "openai",
  "google",
  "alibaba",
  "tencent",
  "bytedance",
  "black",
  "forest",
  "labs",
  "microsoft",
  "xai",
  "recraft",
  "ideogram",
  "leonardo",
]);

export type ImageMatchCandidate = {
  arena_model: string;
  arena_provider: string | null;
  score: number;
};

export type ImageMatchMappedModel = {
  artificial_analysis_slug: string | null;
  artificial_analysis_name: string | null;
  artificial_analysis_provider: string | null;
  best_match: ImageMatchCandidate | null;
  candidates: ImageMatchCandidate[];
};

export type ImageMatchModelMappingPayload = {
  artificial_analysis_fetched_at_epoch_seconds: number | null;
  arena_ai_fetched_at_epoch_seconds: number | null;
  total_artificial_analysis_models: number;
  total_arena_ai_models: number;
  max_candidates: number;
  void_threshold: number | null;
  voided_count: number;
  models: ImageMatchMappedModel[];
};

export type ImageModelsUnionPayload = {
  artificial_analysis_fetched_at_epoch_seconds: number | null;
  arena_ai_fetched_at_epoch_seconds: number | null;
  total_artificial_analysis_models: number;
  total_arena_ai_models: number;
  void_threshold: number | null;
  voided_count: number;
  total_union_models: number;
  models: Record<string, unknown>[];
};

export type ImageMatchModelMappingOptions = {
  maxCandidates?: number;
  useEmbeddings?: boolean;
  embeddingModel?: string;
  embeddingWeight?: number;
};

export type ImageModelsUnionOptions = {
  maxCandidates?: number;
  useEmbeddings?: boolean;
  embeddingModel?: string;
  embeddingWeight?: number;
};

type EmbeddingCachePayload = {
  model: string;
  vectors: Record<string, number[]>;
};

function asRecord(value: unknown): Record<string, unknown> {
  return value != null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function normalizeModelName(value: string): string {
  return (
    value
      .toLowerCase()
      // Preserve bracketed qualifiers as tokens, only strip bracket chars.
      .replace(/[\[\]()]/g, " ")
      .replace(/gpt[\s-]*image/g, "gptimage")
      .replace(/nano[\s-]*banana/g, "nanobanana")
      .replace(/[._:/]+/g, "-")
      .replace(/[^a-z0-9-]+/g, "-")
      .replace(/-+/g, "-")
      .replace(/^-+|-+$/g, "")
  );
}

function embeddingText(value: string): string {
  return normalizeModelName(value).replace(/-/g, " ").trim();
}

function splitTokens(value: string): string[] {
  return normalizeModelName(value)
    .split("-")
    .flatMap((token) => token.split(/(?<=\D)(?=\d)|(?<=\d)(?=\D)/g))
    .filter(Boolean);
}

function providerPrefix(provider: string | null | undefined): string | null {
  if (!provider) {
    return null;
  }
  const left = provider.split("·")[0]?.trim().toLowerCase();
  return left && left.length > 0 ? left : null;
}

function commonPrefixLength(left: string, right: string): number {
  const maxLength = Math.min(left.length, right.length);
  let index = 0;
  while (index < maxLength && left[index] === right[index]) {
    index += 1;
  }
  return index;
}

function toNumericToken(token: string): number | null {
  if (!/^\d+$/.test(token)) {
    return null;
  }
  const numeric = Number(token);
  return Number.isFinite(numeric) ? numeric : null;
}

function tokenSimilarity(left: string, right: string): number {
  if (left === right) {
    return 1;
  }
  const leftNumeric = toNumericToken(left);
  const rightNumeric = toNumericToken(right);
  if (leftNumeric != null && rightNumeric != null) {
    const gap = Math.abs(leftNumeric - rightNumeric);
    return Math.max(0, 1 - gap / Math.max(1, leftNumeric, rightNumeric));
  }
  if (left.includes(right) || right.includes(left)) {
    const shorter = Math.max(1, Math.min(left.length, right.length));
    return Math.min(0.85, shorter / Math.max(left.length, right.length));
  }
  const prefix = commonPrefixLength(left, right);
  if (prefix >= 2) {
    return (prefix / Math.max(1, Math.min(left.length, right.length))) * 0.7;
  }
  return 0;
}

function alignedTokenScore(
  leftTokens: string[],
  rightTokens: string[],
): number {
  const memo = new Map<string, number>();
  function solve(leftIndex: number, rightIndex: number): number {
    const key = `${leftIndex}:${rightIndex}`;
    const cached = memo.get(key);
    if (cached != null) {
      return cached;
    }
    if (leftIndex >= leftTokens.length || rightIndex >= rightTokens.length) {
      memo.set(key, 0);
      return 0;
    }
    const match =
      tokenSimilarity(
        leftTokens[leftIndex] ?? "",
        rightTokens[rightIndex] ?? "",
      ) + solve(leftIndex + 1, rightIndex + 1);
    const skipLeft = solve(leftIndex + 1, rightIndex);
    const skipRight = solve(leftIndex, rightIndex + 1);
    const best = Math.max(match, skipLeft, skipRight);
    memo.set(key, best);
    return best;
  }
  return solve(0, 0);
}

function setJaccard(leftTokens: string[], rightTokens: string[]): number {
  const leftSet = new Set(leftTokens);
  const rightSet = new Set(rightTokens);
  if (leftSet.size === 0 && rightSet.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const token of leftSet) {
    if (rightSet.has(token)) {
      intersection += 1;
    }
  }
  const union = leftSet.size + rightSet.size - intersection;
  return union > 0 ? intersection / union : 0;
}

function positionalExactMatches(
  leftTokens: string[],
  rightTokens: string[],
): number {
  const limit = Math.min(leftTokens.length, rightTokens.length);
  let matches = 0;
  for (let index = 0; index < limit; index += 1) {
    if (leftTokens[index] === rightTokens[index]) {
      matches += 1;
    }
  }
  return matches;
}

function isDistinctiveToken(token: string): boolean {
  return token.length >= 3 && !NOISE_TOKENS.has(token) && !/^\d+$/.test(token);
}

function distinctiveCoverage(
  leftTokens: string[],
  rightTokens: string[],
): number {
  const leftDistinctive = leftTokens.filter((token) =>
    isDistinctiveToken(token),
  );
  if (leftDistinctive.length === 0) {
    return 0;
  }
  const rightSet = new Set(rightTokens);
  const matched = leftDistinctive.filter((token) => rightSet.has(token)).length;
  return matched / leftDistinctive.length;
}

function qualifierSignals(
  leftTokens: string[],
  rightTokens: string[],
): { matchBonus: number; missPenalty: number } {
  const leftQualifiers = leftTokens.filter((token) =>
    QUALIFIER_TOKENS.has(token),
  );
  if (leftQualifiers.length === 0) {
    return { matchBonus: 0, missPenalty: 0 };
  }
  const rightSet = new Set(rightTokens);
  let matched = 0;
  let missed = 0;
  for (const qualifier of leftQualifiers) {
    if (rightSet.has(qualifier)) {
      matched += 1;
    } else {
      missed += 1;
    }
  }
  return {
    matchBonus: matched * QUALIFIER_MATCH_WEIGHT,
    missPenalty: Math.min(
      MAX_QUALIFIER_PENALTY,
      missed * QUALIFIER_MISS_PENALTY,
    ),
  };
}

function computeNameSimilarity(left: string, right: string): number {
  const leftNormalized = normalizeModelName(left);
  const rightNormalized = normalizeModelName(right);
  if (!leftNormalized || !rightNormalized) {
    return 0;
  }

  const leftTokens = splitTokens(left);
  const rightTokens = splitTokens(right);
  const aligned =
    alignedTokenScore(leftTokens, rightTokens) /
    Math.max(1, Math.max(leftTokens.length, rightTokens.length));
  const jaccard = setJaccard(leftTokens, rightTokens);
  const positional =
    positionalExactMatches(leftTokens, rightTokens) /
    Math.max(1, Math.min(leftTokens.length, rightTokens.length));
  const containment =
    leftNormalized.includes(rightNormalized) ||
    rightNormalized.includes(leftNormalized)
      ? 1
      : 0;
  const exact = leftNormalized === rightNormalized ? 1 : 0;
  const coverage = distinctiveCoverage(leftTokens, rightTokens);
  const qualifier = qualifierSignals(leftTokens, rightTokens);
  const leftVersion = leftTokens
    .map((token) => toNumericToken(token))
    .filter((value): value is number => value != null)
    .slice(0, 2);
  const rightVersion = rightTokens
    .map((token) => toNumericToken(token))
    .filter((value): value is number => value != null)
    .slice(0, 2);
  let versionPenalty = 0;
  if (leftVersion.length > 0 && rightVersion.length > 0) {
    if (leftVersion[0] !== rightVersion[0]) {
      versionPenalty += 3;
    } else if (
      leftVersion.length > 1 &&
      rightVersion.length > 1 &&
      leftVersion[1] !== rightVersion[1]
    ) {
      const leftMinor = leftVersion[1] as number;
      const rightMinor = rightVersion[1] as number;
      versionPenalty += Math.min(2, Math.abs(leftMinor - rightMinor) * 0.8);
    }
  }

  const weighted =
    exact * 10 +
    aligned * 8 +
    jaccard * 6 +
    positional * 5 +
    containment * 2 +
    coverage * TOKEN_COVERAGE_WEIGHT +
    qualifier.matchBonus -
    qualifier.missPenalty -
    versionPenalty;
  return Number(weighted.toFixed(4));
}

function getArtificialAnalysisNames(
  model: ArtificialAnalysisImageModel,
): string[] {
  const names: string[] = [];
  if (typeof model.name === "string" && model.name.length > 0) {
    names.push(model.name);
  }
  if (typeof model.slug === "string" && model.slug.length > 0) {
    names.push(model.slug);
  }
  return names.length > 0 ? names : [""];
}

function getArtificialAnalysisEmbeddingText(
  model: ArtificialAnalysisImageModel,
): string {
  const text = [...new Set(getArtificialAnalysisNames(model))]
    .map((name) => embeddingText(name))
    .filter((name) => name.length > 0)
    .join(" | ");
  return text.length > 0 ? text : "unknown";
}

function getArenaEmbeddingText(model: ArenaAiImageModel): string {
  const text = embeddingText(model.model);
  return text.length > 0 ? text : "unknown";
}

function cosineSimilarity(left: number[], right: number[]): number {
  if (left.length === 0 || right.length === 0 || left.length !== right.length) {
    return 0;
  }
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index] ?? 0;
    const rightValue = right[index] ?? 0;
    dot += leftValue * rightValue;
    leftNorm += leftValue * leftValue;
    rightNorm += rightValue * rightValue;
  }
  if (leftNorm === 0 || rightNorm === 0) {
    return 0;
  }
  return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

async function readEmbeddingCache(
  embeddingModel: string,
): Promise<Record<string, number[]>> {
  try {
    const raw = await readFile(EMBEDDING_CACHE_PATH, "utf-8");
    const parsed = JSON.parse(raw) as EmbeddingCachePayload;
    if (parsed.model !== embeddingModel) {
      return {};
    }
    if (!parsed.vectors || typeof parsed.vectors !== "object") {
      return {};
    }
    return parsed.vectors;
  } catch {
    return {};
  }
}

async function writeEmbeddingCache(
  embeddingModel: string,
  vectors: Record<string, number[]>,
): Promise<void> {
  try {
    await mkdir(resolve(".cache"), { recursive: true });
    const payload: EmbeddingCachePayload = {
      model: embeddingModel,
      vectors,
    };
    await writeFile(
      EMBEDDING_CACHE_PATH,
      `${JSON.stringify(payload, null, 2)}\n`,
      "utf-8",
    );
  } catch {
    // Best-effort cache.
  }
}

async function getEmbeddingVectorsByText(
  texts: string[],
  embeddingModel: string,
): Promise<Record<string, number[]>> {
  const uniqueTexts = [...new Set(texts)].filter((text) => text.length > 0);
  if (uniqueTexts.length === 0) {
    return {};
  }

  const cache = await readEmbeddingCache(embeddingModel);
  const missingTexts = uniqueTexts.filter((text) => !cache[text]);
  if (missingTexts.length > 0) {
    const embeddings = OpenRouterEmbeddings({ model: embeddingModel });
    for (
      let index = 0;
      index < missingTexts.length;
      index += EMBEDDING_BATCH_SIZE
    ) {
      const batch = missingTexts.slice(index, index + EMBEDDING_BATCH_SIZE);
      if (batch.length === 0) {
        continue;
      }
      const vectors = await embeddings.embedDocuments(batch);
      for (let vectorIndex = 0; vectorIndex < batch.length; vectorIndex += 1) {
        const text = batch[vectorIndex];
        const vector = vectors[vectorIndex];
        if (text && Array.isArray(vector)) {
          cache[text] = vector;
        }
      }
    }
    await writeEmbeddingCache(embeddingModel, cache);
  }

  return Object.fromEntries(
    uniqueTexts.map((text) => [text, cache[text] ?? []]),
  );
}

async function getEmbeddingBonusByPair(
  artificialAnalysisModels: ArtificialAnalysisImageModel[],
  arenaModels: ArenaAiImageModel[],
  {
    useEmbeddings,
    embeddingModel,
    embeddingWeight,
  }: {
    useEmbeddings: boolean;
    embeddingModel: string;
    embeddingWeight: number;
  },
): Promise<Map<string, number>> {
  const pairBonus = new Map<string, number>();
  if (!useEmbeddings) {
    return pairBonus;
  }
  if (!process.env.OPENROUTER_API_KEY) {
    return pairBonus;
  }

  try {
    const aaTexts = artificialAnalysisModels.map((model) =>
      getArtificialAnalysisEmbeddingText(model),
    );
    const arenaTexts = arenaModels.map((model) => getArenaEmbeddingText(model));
    const vectorsByText = await getEmbeddingVectorsByText(
      [...aaTexts, ...arenaTexts],
      embeddingModel,
    );
    for (let aaIndex = 0; aaIndex < aaTexts.length; aaIndex += 1) {
      const aaVector = vectorsByText[aaTexts[aaIndex] ?? ""] ?? [];
      for (
        let arenaIndex = 0;
        arenaIndex < arenaTexts.length;
        arenaIndex += 1
      ) {
        const arenaVector = vectorsByText[arenaTexts[arenaIndex] ?? ""] ?? [];
        const similarity = Math.max(0, cosineSimilarity(aaVector, arenaVector));
        const bonus = similarity * embeddingWeight;
        pairBonus.set(`${aaIndex}:${arenaIndex}`, Number(bonus.toFixed(4)));
      }
    }
    return pairBonus;
  } catch {
    return pairBonus;
  }
}

function getFamilyAnchorTokens(name: string): string[] {
  const tokens = splitTokens(name).filter(
    (token) =>
      isDistinctiveToken(token) &&
      !QUALIFIER_TOKENS.has(token) &&
      !PROVIDER_NOISE_TOKENS.has(token),
  );
  return [...new Set(tokens)];
}

function hasFamilyAnchorOverlap(
  artificialAnalysisModel: ArtificialAnalysisImageModel,
  arenaModelName: string,
): boolean {
  const aaAnchors = getArtificialAnalysisNames(artificialAnalysisModel).flatMap(
    (name) => getFamilyAnchorTokens(name),
  );
  if (aaAnchors.length === 0) {
    return true;
  }
  const arenaAnchorSet = new Set(getFamilyAnchorTokens(arenaModelName));
  return aaAnchors.some((token) => arenaAnchorSet.has(token));
}

function computeCandidateScore(
  artificialAnalysisModel: ArtificialAnalysisImageModel,
  arenaModel: ArenaAiImageModel,
  artificialAnalysisRank: number | null,
  arenaRank: number | null,
  embeddingBonus = 0,
): number {
  const baseScore = Math.max(
    ...getArtificialAnalysisNames(artificialAnalysisModel).map((name) =>
      computeNameSimilarity(name, arenaModel.model),
    ),
  );
  const modelCreator = asRecord(artificialAnalysisModel.model_creator);
  const aaProvider =
    typeof modelCreator.name === "string"
      ? modelCreator.name.toLowerCase()
      : null;
  const arenaProvider = providerPrefix(arenaModel.provider);
  if (
    aaProvider &&
    arenaProvider &&
    (aaProvider.includes(arenaProvider) || arenaProvider.includes(aaProvider))
  ) {
    let score = baseScore + PROVIDER_MATCH_REWARD;
    if (artificialAnalysisRank != null && arenaRank != null) {
      const gap = Math.abs(artificialAnalysisRank - arenaRank);
      if (gap <= RANK_PROXIMITY_RADIUS) {
        score +=
          ((RANK_PROXIMITY_RADIUS - gap + 1) / (RANK_PROXIMITY_RADIUS + 1)) *
          RANK_PROXIMITY_MAX_BONUS;
      }
    }
    return Number((score + embeddingBonus).toFixed(4));
  }
  let score = baseScore;
  if (artificialAnalysisRank != null && arenaRank != null) {
    const gap = Math.abs(artificialAnalysisRank - arenaRank);
    if (gap <= RANK_PROXIMITY_RADIUS) {
      score +=
        ((RANK_PROXIMITY_RADIUS - gap + 1) / (RANK_PROXIMITY_RADIUS + 1)) *
        RANK_PROXIMITY_MAX_BONUS;
    }
  }
  return Number((score + embeddingBonus).toFixed(4));
}

function isAcceptedBestCandidate(candidates: ImageMatchCandidate[]): boolean {
  const best = candidates[0];
  if (!best || best.score < MIN_ACCEPTED_CANDIDATE_SCORE) {
    return false;
  }
  const second = candidates[1];
  if (!second) {
    return true;
  }
  // Avoid accepting near-random ties, but keep strong-family matches.
  if (best.score - second.score < 0.75 && best.score < 9) {
    return false;
  }
  return true;
}

function isAcceptedBestCandidateForRank(
  artificialAnalysisModel: ArtificialAnalysisImageModel,
  candidates: ImageMatchCandidate[],
  artificialAnalysisRank: number | null,
): boolean {
  const best = candidates[0];
  if (
    best &&
    !hasFamilyAnchorOverlap(artificialAnalysisModel, best.arena_model)
  ) {
    return false;
  }
  if (isAcceptedBestCandidate(candidates)) {
    return true;
  }
  if (
    artificialAnalysisRank != null &&
    artificialAnalysisRank <= TOP_RANK_PROTECTION_COUNT
  ) {
    const best = candidates[0];
    const second = candidates[1];
    if (!best || best.score < MIN_ACCEPTED_CANDIDATE_SCORE) {
      return false;
    }
    const margin =
      best && second ? best.score - second.score : TOP_RANK_PROTECTION_MARGIN;
    if (margin >= TOP_RANK_PROTECTION_MARGIN) {
      return true;
    }
  }
  return false;
}

function applyDynamicVoid<
  T extends {
    best_match: ImageMatchCandidate | null;
    candidates?: ImageMatchCandidate[];
  },
>(models: T[]): { threshold: number | null; voided: number } {
  const scores = models
    .map((model) => model.best_match?.score)
    .filter((score): score is number => score != null)
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
    const topCandidate = model.candidates?.[0];
    const secondCandidate = model.candidates?.[1];
    const margin =
      topCandidate && secondCandidate
        ? topCandidate.score - secondCandidate.score
        : null;
    const rowIndex = models.indexOf(model);
    const isProtectedTopRank =
      rowIndex < TOP_RANK_PROTECTION_COUNT &&
      score != null &&
      score >= threshold - TOP_RANK_PROTECTION_THRESHOLD_DELTA &&
      (margin == null || margin >= TOP_RANK_PROTECTION_MARGIN);

    if (isProtectedTopRank) {
      continue;
    }
    if (score != null && score < threshold) {
      model.best_match = null;
      voided += 1;
    }
  }
  return { threshold, voided };
}

function mapModel(
  artificialAnalysisModel: ArtificialAnalysisImageModel,
  arenaModels: ArenaAiImageModel[],
  maxCandidates: number,
  artificialAnalysisRank: number | null,
  embeddingBonusByPair: Map<string, number>,
  artificialAnalysisIndex: number,
): ImageMatchMappedModel {
  const scoredCandidates = arenaModels
    .map((arenaModel, arenaIndex) => ({
      arena_model: arenaModel.model,
      arena_provider: arenaModel.provider,
      score: computeCandidateScore(
        artificialAnalysisModel,
        arenaModel,
        artificialAnalysisRank,
        arenaIndex + 1,
        embeddingBonusByPair.get(`${artificialAnalysisIndex}:${arenaIndex}`) ??
          0,
      ),
    }))
    .sort((left, right) => right.score - left.score);

  const topCandidates = scoredCandidates.slice(0, maxCandidates);
  const bestCandidate = isAcceptedBestCandidateForRank(
    artificialAnalysisModel,
    topCandidates,
    artificialAnalysisRank,
  )
    ? (topCandidates[0] ?? null)
    : null;

  return {
    artificial_analysis_slug:
      typeof artificialAnalysisModel.slug === "string"
        ? artificialAnalysisModel.slug
        : null,
    artificial_analysis_name:
      typeof artificialAnalysisModel.name === "string"
        ? artificialAnalysisModel.name
        : null,
    artificial_analysis_provider:
      typeof asRecord(artificialAnalysisModel.model_creator).name === "string"
        ? (asRecord(artificialAnalysisModel.model_creator).name as string)
        : null,
    best_match: bestCandidate,
    candidates: topCandidates,
  };
}

export async function getImageMatchModelMapping(
  options: ImageMatchModelMappingOptions = {},
): Promise<ImageMatchModelMappingPayload> {
  const maxCandidates = options.maxCandidates ?? DEFAULT_MAX_CANDIDATES;
  const useEmbeddings = options.useEmbeddings ?? false;
  const embeddingModel = options.embeddingModel ?? DEFAULT_EMBEDDING_MODEL;
  const embeddingWeight = options.embeddingWeight ?? DEFAULT_EMBEDDING_WEIGHT;
  const [artificialAnalysisPayload, arenaPayload] = await Promise.all([
    getArtificialAnalysisImageStats(),
    getArenaAiImageStats(),
  ]);
  const artificialAnalysisModels = artificialAnalysisPayload.data ?? [];
  const arenaModels = arenaPayload.rows ?? [];
  const embeddingBonusByPair = await getEmbeddingBonusByPair(
    artificialAnalysisModels,
    arenaModels,
    {
      useEmbeddings,
      embeddingModel,
      embeddingWeight,
    },
  );
  const models = artificialAnalysisModels.map((model, index) =>
    mapModel(
      model,
      arenaModels,
      maxCandidates,
      index + 1,
      embeddingBonusByPair,
      index,
    ),
  );
  const voidStats = applyDynamicVoid(models);

  return {
    artificial_analysis_fetched_at_epoch_seconds:
      artificialAnalysisPayload.fetched_at_epoch_seconds,
    arena_ai_fetched_at_epoch_seconds: arenaPayload.fetched_at_epoch_seconds,
    total_artificial_analysis_models: artificialAnalysisModels.length,
    total_arena_ai_models: arenaModels.length,
    max_candidates: maxCandidates,
    void_threshold: voidStats.threshold,
    voided_count: voidStats.voided,
    models,
  };
}

function mergeRows(
  mappedModel: ImageMatchMappedModel,
  artificialAnalysisModelsBySlug: Map<string, ArtificialAnalysisImageModel>,
  arenaModelsByName: Map<string, ArenaAiImageModel>,
): Record<string, unknown> {
  const artificialAnalysis =
    mappedModel.artificial_analysis_slug != null
      ? (artificialAnalysisModelsBySlug.get(
          mappedModel.artificial_analysis_slug,
        ) ?? null)
      : null;
  const arena =
    mappedModel.best_match?.arena_model != null
      ? (arenaModelsByName.get(mappedModel.best_match.arena_model) ?? null)
      : null;

  return {
    artificial_analysis_slug: mappedModel.artificial_analysis_slug,
    artificial_analysis_name: mappedModel.artificial_analysis_name,
    artificial_analysis_provider: mappedModel.artificial_analysis_provider,
    best_match: mappedModel.best_match,
    artificial_analysis: artificialAnalysis,
    arena_ai: arena,
    union: {
      ...(arena ?? {}),
      ...(artificialAnalysis ?? {}),
    },
  };
}

export async function getImageModelsUnion(
  options: ImageModelsUnionOptions = {},
): Promise<ImageModelsUnionPayload> {
  const mappingOptions: ImageMatchModelMappingOptions = {
    ...(options.maxCandidates != null
      ? { maxCandidates: options.maxCandidates }
      : {}),
    ...(options.useEmbeddings != null
      ? { useEmbeddings: options.useEmbeddings }
      : {}),
    ...(options.embeddingModel != null
      ? { embeddingModel: options.embeddingModel }
      : {}),
    ...(options.embeddingWeight != null
      ? { embeddingWeight: options.embeddingWeight }
      : {}),
  };
  const mapping =
    options.maxCandidates != null
      ? await getImageMatchModelMapping(mappingOptions)
      : await getImageMatchModelMapping(mappingOptions);
  const [artificialAnalysisPayload, arenaPayload] = await Promise.all([
    getArtificialAnalysisImageStats(),
    getArenaAiImageStats(),
  ]);
  const artificialAnalysisModels = artificialAnalysisPayload.data ?? [];
  const arenaModels = arenaPayload.rows ?? [];
  const artificialAnalysisModelsBySlug = new Map(
    artificialAnalysisModels
      .filter((model) => typeof model.slug === "string")
      .map((model) => [model.slug as string, model]),
  );
  const arenaModelsByName = new Map(
    arenaModels.map((model) => [model.model, model]),
  );

  const mappedRows = mapping.models.map((model) =>
    mergeRows(model, artificialAnalysisModelsBySlug, arenaModelsByName),
  );

  const matchedArenaNames = new Set(
    mapping.models
      .map((model) => model.best_match?.arena_model)
      .filter((value): value is string => Boolean(value)),
  );
  const unmatchedArenaRows = arenaModels
    .filter((model) => !matchedArenaNames.has(model.model))
    .map((model) => ({
      artificial_analysis_slug: null,
      artificial_analysis_name: null,
      artificial_analysis_provider: null,
      best_match: null,
      artificial_analysis: null,
      arena_ai: model,
      union: { ...model },
    }));

  return {
    artificial_analysis_fetched_at_epoch_seconds:
      mapping.artificial_analysis_fetched_at_epoch_seconds,
    arena_ai_fetched_at_epoch_seconds:
      mapping.arena_ai_fetched_at_epoch_seconds,
    total_artificial_analysis_models: mapping.total_artificial_analysis_models,
    total_arena_ai_models: mapping.total_arena_ai_models,
    void_threshold: mapping.void_threshold,
    voided_count: mapping.voided_count,
    total_union_models: mappedRows.length + unmatchedArenaRows.length,
    models: [...mappedRows, ...unmatchedArenaRows],
  };
}
