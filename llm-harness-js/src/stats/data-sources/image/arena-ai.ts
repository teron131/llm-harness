import { fetchWithTimeout, nowEpochSeconds, percentileRank } from "../utils";

const ARENA_AI_BASE_URL = "https://arena.ai/leaderboard/text-to-image";
const REQUEST_TIMEOUT_MS = 30_000;
const MIN_VALID_ROWS = 20;
const MIN_VALID_CATEGORIES = 4;

export const ARENA_AI_DEFAULT_CATEGORY_SLUGS = [
  "commercial-design",
  "3d-modeling",
  "cartoon",
  "photorealistic",
  "art",
  "portraits",
  "text-rendering",
] as const;

type ArenaAiImageGroupedCategoryName =
  | "photorealistic"
  | "illustrative"
  | "contextual";

export const ARENA_AI_GROUPED_CATEGORY_SLUGS: Record<
  ArenaAiImageGroupedCategoryName,
  readonly string[]
> = {
  photorealistic: ["photorealistic", "portraits"],
  illustrative: ["cartoon", "art"],
  contextual: ["commercial-design", "3d-modeling", "text-rendering"],
};

const ARENA_AI_GROUPED_CATEGORY_NAMES: readonly ArenaAiImageGroupedCategoryName[] =
  ["photorealistic", "illustrative", "contextual"];

const ARENA_AI_GROUP_BY_SLUG = new Map<string, ArenaAiImageGroupedCategoryName>(
  ARENA_AI_GROUPED_CATEGORY_NAMES.flatMap((groupName) =>
    ARENA_AI_GROUPED_CATEGORY_SLUGS[groupName].map((slug) => [slug, groupName]),
  ),
);

type NumberOrNull = number | null;

export type ArenaAiImageRow = {
  model: string;
  provider: string | null;
  score: NumberOrNull;
  ci95: string | null;
  votes: NumberOrNull;
};

export type ArenaAiImageCategoryPayload = {
  fetched_at_epoch_seconds: number;
  category_slug: string;
  source_url: string;
  final_url: string | null;
  status_code: number | null;
  challenge_detected: boolean;
  rows_with_score: number;
  rows: ArenaAiImageRow[];
};

export type ArenaAiImageAggregatedCategoryRow = {
  rank: number;
  score: NumberOrNull;
  ci95: string | null;
  votes: NumberOrNull;
};

export type ArenaAiImageAggregatedModel = {
  model: string;
  provider: string | null;
  category_count: number;
  average_score: NumberOrNull;
  vote_weighted_score: NumberOrNull;
  average_rank: NumberOrNull;
  votes_sum: number;
  categories: Record<string, ArenaAiImageAggregatedCategoryRow>;
  weighted_scores: {
    photorealistic: NumberOrNull;
    illustrative: NumberOrNull;
    contextual: NumberOrNull;
    grouped_overall: NumberOrNull;
  };
  grouped_votes: {
    photorealistic: number;
    illustrative: number;
    contextual: number;
    total: number;
  };
  percentiles: {
    vote_weighted_percentile: NumberOrNull;
    photorealistic_percentile: NumberOrNull;
    illustrative_percentile: NumberOrNull;
    contextual_percentile: NumberOrNull;
    grouped_overall_percentile: NumberOrNull;
  };
};

/**
 * Arena text-to-image aggregated response.
 *
 * This source is failure-safe by design and returns an empty payload when all
 * category fetches fail.
 */
export type ArenaAiImageOutputPayload = {
  fetched_at_epoch_seconds: number;
  base_url: string;
  category_slugs: string[];
  categories: ArenaAiImageCategoryPayload[];
  grouped_category_slugs: Record<
    ArenaAiImageGroupedCategoryName,
    readonly string[]
  >;
  valid_categories: string[];
  total_valid_categories: number;
  total_models_aggregated: number;
  scrape_feasible_now: boolean;
  rows: ArenaAiImageAggregatedModel[];
};

/**
 * Arena source options.
 */
export type ArenaAiImageOptions = {
  categorySlugs?: string[];
  minValidRows?: number;
  minValidCategories?: number;
};

type ArenaAiImageAggregateAccumulator = {
  model: string;
  provider: string | null;
  category_rows: Record<string, ArenaAiImageAggregatedCategoryRow>;
  category_count: number;
  votes_sum: number;
  score_sum: number;
  score_weighted_sum: number;
  rank_sum: number;
};

type ArenaAiImageGroupedAccumulatorValue = {
  score_weighted_sum: number;
  votes_sum: number;
  score_sum: number;
  count: number;
};

type ArenaAiImageGroupedAccumulator = Record<
  ArenaAiImageGroupedCategoryName,
  ArenaAiImageGroupedAccumulatorValue
>;

function detectChallenge(html: string): boolean {
  return /challenge-platform|__CF\$cv\$params|Verify you are human|Security Verification/i.test(
    html,
  );
}

function asFiniteNumber(value: unknown): NumberOrNull {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function extractLeaderboardRows(html: string): ArenaAiImageRow[] {
  const titleMatches = [...html.matchAll(/<a[^>]*\btitle="([^"]+)"/g)];
  const rows: ArenaAiImageRow[] = [];

  for (let index = 0; index < titleMatches.length; index += 1) {
    const title = titleMatches[index]?.[1] ?? "";
    const start = titleMatches[index]?.index ?? 0;
    const end =
      titleMatches[index + 1]?.index ?? Math.min(start + 4_000, html.length);
    const rowBlock = html.slice(start, end);
    const scoreMatch = rowBlock.match(
      />(\d{3,4})<\/span><span[^>]*>±(\d+)<\/span>/,
    );
    const votesMatch = rowBlock.match(
      /±\d+<\/span>(?:.|\n){0,1200}?>([\d,]{2,})<\/span>/,
    );
    const providerMatch = rowBlock.match(/>([^<]+ · [^<]+)<\/span>/);

    rows.push({
      model: title,
      provider: providerMatch?.[1] ?? null,
      score: asFiniteNumber(scoreMatch?.[1]),
      ci95: scoreMatch?.[2] ? `±${scoreMatch[2]}` : null,
      votes: votesMatch?.[1]
        ? asFiniteNumber(votesMatch[1].replaceAll(",", ""))
        : null,
    });
  }

  return rows;
}

async function fetchCategory(
  baseUrl: string,
  categorySlug: string,
): Promise<ArenaAiImageCategoryPayload> {
  const sourceUrl = `${baseUrl}/${categorySlug}`;
  try {
    const response = await fetchWithTimeout(
      sourceUrl,
      {
        headers: {
          "user-agent":
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
          "accept-language": "en-US,en;q=0.9",
        },
        redirect: "follow",
      },
      REQUEST_TIMEOUT_MS,
    );

    const html = await response.text();
    const rowsWithScore = extractLeaderboardRows(html).filter(
      (row) => row.score != null,
    );

    return {
      fetched_at_epoch_seconds: nowEpochSeconds(),
      category_slug: categorySlug,
      source_url: sourceUrl,
      final_url: response.url,
      status_code: response.status,
      challenge_detected: detectChallenge(html),
      rows_with_score: rowsWithScore.length,
      rows: rowsWithScore,
    };
  } catch {
    return {
      fetched_at_epoch_seconds: nowEpochSeconds(),
      category_slug: categorySlug,
      source_url: sourceUrl,
      final_url: null,
      status_code: null,
      challenge_detected: false,
      rows_with_score: 0,
      rows: [],
    };
  }
}

function roundTo4(value: number): number {
  return Number(value.toFixed(4));
}

function weightedScoreOrAverage(
  weightedSum: number,
  votesSum: number,
  scoreSum: number,
  count: number,
): NumberOrNull {
  if (votesSum > 0) {
    return roundTo4(weightedSum / votesSum);
  }
  if (count > 0) {
    return roundTo4(scoreSum / count);
  }
  return null;
}

function detectGroupedCategory(
  categorySlug: string,
): ArenaAiImageGroupedCategoryName | null {
  return ARENA_AI_GROUP_BY_SLUG.get(categorySlug) ?? null;
}

function createGroupedAccumulator(): ArenaAiImageGroupedAccumulator {
  return {
    photorealistic: {
      score_weighted_sum: 0,
      votes_sum: 0,
      score_sum: 0,
      count: 0,
    },
    illustrative: {
      score_weighted_sum: 0,
      votes_sum: 0,
      score_sum: 0,
      count: 0,
    },
    contextual: {
      score_weighted_sum: 0,
      votes_sum: 0,
      score_sum: 0,
      count: 0,
    },
  };
}

function buildGroupedScores(
  categoryRows: Record<string, ArenaAiImageAggregatedCategoryRow>,
): Pick<ArenaAiImageAggregatedModel, "weighted_scores" | "grouped_votes"> {
  const groupedAccumulator = createGroupedAccumulator();
  for (const [categorySlug, row] of Object.entries(categoryRows)) {
    const groupedCategory = detectGroupedCategory(categorySlug);
    if (!groupedCategory) {
      continue;
    }
    const score = row.score ?? 0;
    const votes = row.votes ?? 0;
    groupedAccumulator[groupedCategory].score_weighted_sum += score * votes;
    groupedAccumulator[groupedCategory].votes_sum += votes;
    groupedAccumulator[groupedCategory].score_sum += score;
    groupedAccumulator[groupedCategory].count += 1;
  }

  const groupedScores = {
    photorealistic: weightedScoreOrAverage(
      groupedAccumulator.photorealistic.score_weighted_sum,
      groupedAccumulator.photorealistic.votes_sum,
      groupedAccumulator.photorealistic.score_sum,
      groupedAccumulator.photorealistic.count,
    ),
    illustrative: weightedScoreOrAverage(
      groupedAccumulator.illustrative.score_weighted_sum,
      groupedAccumulator.illustrative.votes_sum,
      groupedAccumulator.illustrative.score_sum,
      groupedAccumulator.illustrative.count,
    ),
    contextual: weightedScoreOrAverage(
      groupedAccumulator.contextual.score_weighted_sum,
      groupedAccumulator.contextual.votes_sum,
      groupedAccumulator.contextual.score_sum,
      groupedAccumulator.contextual.count,
    ),
    grouped_overall: null as NumberOrNull,
  };
  const groupedTotalVotes = ARENA_AI_GROUPED_CATEGORY_NAMES.reduce(
    (sum, groupName) => sum + groupedAccumulator[groupName].votes_sum,
    0,
  );
  groupedScores.grouped_overall = weightedScoreOrAverage(
    ARENA_AI_GROUPED_CATEGORY_NAMES.reduce(
      (sum, groupName) =>
        sum + groupedAccumulator[groupName].score_weighted_sum,
      0,
    ),
    groupedTotalVotes,
    ARENA_AI_GROUPED_CATEGORY_NAMES.reduce(
      (sum, groupName) => sum + groupedAccumulator[groupName].score_sum,
      0,
    ),
    ARENA_AI_GROUPED_CATEGORY_NAMES.reduce(
      (sum, groupName) => sum + groupedAccumulator[groupName].count,
      0,
    ),
  );

  return {
    weighted_scores: groupedScores,
    grouped_votes: {
      photorealistic: groupedAccumulator.photorealistic.votes_sum,
      illustrative: groupedAccumulator.illustrative.votes_sum,
      contextual: groupedAccumulator.contextual.votes_sum,
      total: groupedTotalVotes,
    },
  };
}

function buildAggregatedRows(
  categoryPayloads: ArenaAiImageCategoryPayload[],
): ArenaAiImageAggregatedModel[] {
  const byModel = new Map<string, ArenaAiImageAggregateAccumulator>();

  for (const payload of categoryPayloads) {
    for (let index = 0; index < payload.rows.length; index += 1) {
      const row = payload.rows[index];
      if (row == null) {
        continue;
      }
      const existing = byModel.get(row.model);
      const aggregate = existing ?? {
        model: row.model,
        provider: row.provider,
        category_rows: {},
        category_count: 0,
        votes_sum: 0,
        score_sum: 0,
        score_weighted_sum: 0,
        rank_sum: 0,
      };

      aggregate.category_rows[payload.category_slug] = {
        rank: index + 1,
        score: row.score,
        ci95: row.ci95,
        votes: row.votes,
      };
      aggregate.category_count += 1;
      aggregate.rank_sum += index + 1;
      aggregate.score_sum += row.score ?? 0;
      const votes = row.votes ?? 0;
      aggregate.votes_sum += votes;
      aggregate.score_weighted_sum += (row.score ?? 0) * votes;

      if (!existing) {
        byModel.set(row.model, aggregate);
      }
    }
  }

  const aggregatedRows: ArenaAiImageAggregatedModel[] = [];
  for (const aggregate of byModel.values()) {
    const averageScore =
      aggregate.category_count > 0
        ? roundTo4(aggregate.score_sum / aggregate.category_count)
        : null;
    const voteWeightedScore =
      aggregate.votes_sum > 0
        ? roundTo4(aggregate.score_weighted_sum / aggregate.votes_sum)
        : averageScore;
    const averageRank =
      aggregate.category_count > 0
        ? roundTo4(aggregate.rank_sum / aggregate.category_count)
        : null;
    const grouped = buildGroupedScores(aggregate.category_rows);

    aggregatedRows.push({
      model: aggregate.model,
      provider: aggregate.provider,
      category_count: aggregate.category_count,
      average_score: averageScore,
      vote_weighted_score: voteWeightedScore,
      average_rank: averageRank,
      votes_sum: aggregate.votes_sum,
      categories: aggregate.category_rows,
      weighted_scores: grouped.weighted_scores,
      grouped_votes: grouped.grouped_votes,
      percentiles: {
        vote_weighted_percentile: null,
        photorealistic_percentile: null,
        illustrative_percentile: null,
        contextual_percentile: null,
        grouped_overall_percentile: null,
      },
    });
  }

  aggregatedRows.sort((left, right) => {
    const leftScore = left.vote_weighted_score;
    const rightScore = right.vote_weighted_score;
    if (leftScore != null && rightScore != null && rightScore !== leftScore) {
      return rightScore - leftScore;
    }
    if (leftScore == null && rightScore != null) {
      return 1;
    }
    if (leftScore != null && rightScore == null) {
      return -1;
    }
    return right.category_count - left.category_count;
  });

  const voteWeightedValues = aggregatedRows.map(
    (row) => row.vote_weighted_score,
  );
  const photorealisticValues = aggregatedRows.map(
    (row) => row.weighted_scores.photorealistic,
  );
  const illustrativeValues = aggregatedRows.map(
    (row) => row.weighted_scores.illustrative,
  );
  const contextualValues = aggregatedRows.map(
    (row) => row.weighted_scores.contextual,
  );
  const groupedOverallValues = aggregatedRows.map(
    (row) => row.weighted_scores.grouped_overall,
  );

  return aggregatedRows.map((row) => ({
    ...row,
    percentiles: {
      vote_weighted_percentile: percentileRank(
        voteWeightedValues,
        row.vote_weighted_score,
      ),
      photorealistic_percentile: percentileRank(
        photorealisticValues,
        row.weighted_scores.photorealistic,
      ),
      illustrative_percentile: percentileRank(
        illustrativeValues,
        row.weighted_scores.illustrative,
      ),
      contextual_percentile: percentileRank(
        contextualValues,
        row.weighted_scores.contextual,
      ),
      grouped_overall_percentile: percentileRank(
        groupedOverallValues,
        row.weighted_scores.grouped_overall,
      ),
    },
  }));
}

/**
 * Fetch and aggregate Arena text-to-image leaderboard categories.
 */
export async function getArenaAiImageStats(
  options: ArenaAiImageOptions = {},
): Promise<ArenaAiImageOutputPayload> {
  const categorySlugs =
    options.categorySlugs != null && options.categorySlugs.length > 0
      ? options.categorySlugs
      : [...ARENA_AI_DEFAULT_CATEGORY_SLUGS];
  const minValidRows = options.minValidRows ?? MIN_VALID_ROWS;
  const minValidCategories = options.minValidCategories ?? MIN_VALID_CATEGORIES;

  const categories = await Promise.all(
    categorySlugs.map((slug) => fetchCategory(ARENA_AI_BASE_URL, slug)),
  );
  const validCategories = categories.filter(
    (category) => category.rows_with_score >= minValidRows,
  );
  const rows = buildAggregatedRows(validCategories);

  return {
    fetched_at_epoch_seconds: nowEpochSeconds(),
    base_url: ARENA_AI_BASE_URL,
    category_slugs: categorySlugs,
    categories,
    grouped_category_slugs: ARENA_AI_GROUPED_CATEGORY_SLUGS,
    valid_categories: validCategories.map((category) => category.category_slug),
    total_valid_categories: validCategories.length,
    total_models_aggregated: rows.length,
    scrape_feasible_now: validCategories.length >= minValidCategories,
    rows,
  };
}
