const TEXT_TO_IMAGE_URL =
  "https://artificialanalysis.ai/api/v2/data/media/text-to-image?include_categories=true";
const REQUEST_TIMEOUT_MS = 30_000;
const DEFAULT_MIN_MODEL_AGE_DAYS = 365;

type NumberOrNull = number | null;
type GroupName = "Photorealistic" | "Illustrative" | "Contextual";

export const ARTIFICIAL_ANALYSIS_IMAGE_GROUPS: Record<
  GroupName,
  readonly string[]
> = {
  Photorealistic: [
    "General & Photorealistic",
    "People: Portraits",
    "Physical Spaces",
    "Nature & Landscapes",
    "Vintage & Retro",
    "People: Groups & Activities",
  ],
  Illustrative: [
    "Cartoon & Illustration",
    "Anime",
    "Futuristic & Sci-Fi",
    "Fantasy & Mythical",
    "Traditional Art",
  ],
  Contextual: [
    "Text & Typography",
    "UI/UX Design",
    "Commercial",
    "Graphic Design & Digital Rendering",
  ],
};

const GROUP_NAMES: GroupName[] = [
  "Photorealistic",
  "Illustrative",
  "Contextual",
];

type RawCategory = {
  style_category?: string;
  subject_matter_category?: string;
  elo?: number;
  appearances?: number;
  [key: string]: unknown;
};

export type ArtificialAnalysisImageModel = {
  release_date?: string;
  categories?: RawCategory[];
  [key: string]: unknown;
};

type Aggregator = Record<
  GroupName,
  {
    weightedEloSum: number;
    appearanceSum: number;
  }
>;

export type ArtificialAnalysisImagePercentiles = {
  photorealistic_percentile: NumberOrNull;
  illustrative_percentile: NumberOrNull;
  contextual_percentile: NumberOrNull;
  grouped_overall_percentile: NumberOrNull;
};

export type ArtificialAnalysisImageEnrichedModel =
  ArtificialAnalysisImageModel & {
    frequency_weighted_scores: {
      photorealistic: NumberOrNull;
      illustrative: NumberOrNull;
      contextual: NumberOrNull;
      grouped_overall: NumberOrNull;
    };
    aggregated_frequencies: {
      appearances: {
        photorealistic: number;
        illustrative: number;
        contextual: number;
      };
      total_known_appearances: number;
    };
    percentiles: ArtificialAnalysisImagePercentiles;
  };

type RawTextToImagePayload = {
  include_categories?: boolean;
  data?: ArtificialAnalysisImageModel[];
  [key: string]: unknown;
};

/**
 * Artificial Analysis text-to-image source options.
 */
export type ArtificialAnalysisImageOptions = {
  apiKey?: string;
  minModelAgeDays?: number;
};

/**
 * Enriched Artificial Analysis text-to-image payload.
 *
 * On failure, this returns an empty payload with `null` metadata fields.
 */
export type ArtificialAnalysisImageOutputPayload = {
  fetched_at_epoch_seconds: number | null;
  status_code: number | null;
  endpoint: string;
  filter: {
    release_date_excluded_if_older_than_days: number;
    total_models_before_filter: number;
    total_models_after_filter: number;
  };
  grouping_version: "v1";
  grouped_taxonomy: Record<GroupName, readonly string[]>;
  global_aggregates: {
    frequency_weighted_scores: {
      photorealistic: NumberOrNull;
      illustrative: NumberOrNull;
      contextual: NumberOrNull;
      grouped_overall: NumberOrNull;
    };
    aggregated_frequencies: {
      appearances: {
        photorealistic: number;
        illustrative: number;
        contextual: number;
      };
      total_known_appearances: number;
    };
  };
  data: ArtificialAnalysisImageEnrichedModel[];
};

function nowEpochSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

function finiteNumbers(values: unknown[]): number[] {
  return values
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
}

function percentileRank(values: unknown[], value: unknown): NumberOrNull {
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
  return Number(((lessOrEqualCount / finiteValues.length) * 100).toFixed(4));
}

function parseReleaseDateToUtc(releaseDate: string | undefined): number | null {
  if (!releaseDate) {
    return null;
  }
  if (/^\d{4}-\d{2}$/.test(releaseDate)) {
    const [yearText, monthText] = releaseDate.split("-");
    const year = Number(yearText);
    const month = Number(monthText);
    if (!Number.isFinite(year) || !Number.isFinite(month)) {
      return null;
    }
    return Date.UTC(year, month - 1, 1);
  }
  if (/^\d{4}-\d{2}-\d{2}$/.test(releaseDate)) {
    const parsedMs = Date.parse(`${releaseDate}T00:00:00.000Z`);
    return Number.isFinite(parsedMs) ? parsedMs : null;
  }
  return null;
}

function isOlderThanDays(
  releaseDate: string | undefined,
  minAgeDays: number,
): boolean {
  const releaseMs = parseReleaseDateToUtc(releaseDate);
  if (releaseMs == null) {
    return false;
  }
  const ageMs = Date.now() - releaseMs;
  const minAgeMs = minAgeDays * 24 * 60 * 60 * 1000;
  return ageMs > minAgeMs;
}

function detectGroup(category: RawCategory): GroupName | null {
  const style =
    typeof category.style_category === "string"
      ? category.style_category
      : null;
  const subject =
    typeof category.subject_matter_category === "string"
      ? category.subject_matter_category
      : null;

  for (const groupName of GROUP_NAMES) {
    const labels = ARTIFICIAL_ANALYSIS_IMAGE_GROUPS[groupName];
    if (
      (style && labels.includes(style)) ||
      (subject && labels.includes(subject))
    ) {
      return groupName;
    }
  }
  return null;
}

function initAccumulator(): Aggregator {
  return {
    Photorealistic: { weightedEloSum: 0, appearanceSum: 0 },
    Illustrative: { weightedEloSum: 0, appearanceSum: 0 },
    Contextual: { weightedEloSum: 0, appearanceSum: 0 },
  };
}

function frequencyWeightedElo(
  weightedEloSum: number,
  appearanceSum: number,
): NumberOrNull {
  if (!Number.isFinite(weightedEloSum) || !Number.isFinite(appearanceSum)) {
    return null;
  }
  if (appearanceSum <= 0) {
    return null;
  }
  return Number((weightedEloSum / appearanceSum).toFixed(4));
}

function toAggregatedFields(accumulator: Aggregator) {
  const photorealistic = frequencyWeightedElo(
    accumulator.Photorealistic.weightedEloSum,
    accumulator.Photorealistic.appearanceSum,
  );
  const illustrative = frequencyWeightedElo(
    accumulator.Illustrative.weightedEloSum,
    accumulator.Illustrative.appearanceSum,
  );
  const contextual = frequencyWeightedElo(
    accumulator.Contextual.weightedEloSum,
    accumulator.Contextual.appearanceSum,
  );
  const totalKnownAppearances =
    accumulator.Photorealistic.appearanceSum +
    accumulator.Illustrative.appearanceSum +
    accumulator.Contextual.appearanceSum;
  const groupedOverall = frequencyWeightedElo(
    accumulator.Photorealistic.weightedEloSum +
      accumulator.Illustrative.weightedEloSum +
      accumulator.Contextual.weightedEloSum,
    totalKnownAppearances,
  );

  return {
    frequency_weighted_scores: {
      photorealistic,
      illustrative,
      contextual,
      grouped_overall: groupedOverall,
    },
    aggregated_frequencies: {
      appearances: {
        photorealistic: accumulator.Photorealistic.appearanceSum,
        illustrative: accumulator.Illustrative.appearanceSum,
        contextual: accumulator.Contextual.appearanceSum,
      },
      total_known_appearances: totalKnownAppearances,
    },
  };
}

function enrichPayload(
  rawPayload: RawTextToImagePayload,
  minModelAgeDays: number,
): Omit<
  ArtificialAnalysisImageOutputPayload,
  "fetched_at_epoch_seconds" | "status_code" | "endpoint"
> {
  const allModels = Array.isArray(rawPayload.data) ? rawPayload.data : [];
  const models = allModels.filter(
    (model) => !isOlderThanDays(model.release_date, minModelAgeDays),
  );
  const globalAccumulator = initAccumulator();

  const enrichedModels: ArtificialAnalysisImageEnrichedModel[] = models.map(
    (model) => {
      const localAccumulator = initAccumulator();
      const categories = Array.isArray(model.categories)
        ? model.categories
        : [];

      for (const category of categories) {
        const group = detectGroup(category);
        if (!group) {
          continue;
        }
        const elo = Number(category.elo);
        const appearances = Number(category.appearances);
        if (
          !Number.isFinite(elo) ||
          !Number.isFinite(appearances) ||
          appearances <= 0
        ) {
          continue;
        }
        localAccumulator[group].weightedEloSum += elo * appearances;
        localAccumulator[group].appearanceSum += appearances;
        globalAccumulator[group].weightedEloSum += elo * appearances;
        globalAccumulator[group].appearanceSum += appearances;
      }

      return {
        ...model,
        ...toAggregatedFields(localAccumulator),
        percentiles: {
          photorealistic_percentile: null,
          illustrative_percentile: null,
          contextual_percentile: null,
          grouped_overall_percentile: null,
        },
      };
    },
  );

  enrichedModels.sort(
    (left, right) =>
      (right.frequency_weighted_scores.grouped_overall ??
        Number.NEGATIVE_INFINITY) -
      (left.frequency_weighted_scores.grouped_overall ??
        Number.NEGATIVE_INFINITY),
  );

  const photorealisticValues = enrichedModels.map(
    (model) => model.frequency_weighted_scores.photorealistic,
  );
  const illustrativeValues = enrichedModels.map(
    (model) => model.frequency_weighted_scores.illustrative,
  );
  const contextualValues = enrichedModels.map(
    (model) => model.frequency_weighted_scores.contextual,
  );
  const groupedOverallValues = enrichedModels.map(
    (model) => model.frequency_weighted_scores.grouped_overall,
  );

  const data = enrichedModels.map((model) => ({
    ...model,
    percentiles: {
      photorealistic_percentile: percentileRank(
        photorealisticValues,
        model.frequency_weighted_scores.photorealistic,
      ),
      illustrative_percentile: percentileRank(
        illustrativeValues,
        model.frequency_weighted_scores.illustrative,
      ),
      contextual_percentile: percentileRank(
        contextualValues,
        model.frequency_weighted_scores.contextual,
      ),
      grouped_overall_percentile: percentileRank(
        groupedOverallValues,
        model.frequency_weighted_scores.grouped_overall,
      ),
    },
  }));

  return {
    filter: {
      release_date_excluded_if_older_than_days: minModelAgeDays,
      total_models_before_filter: allModels.length,
      total_models_after_filter: models.length,
    },
    grouping_version: "v1",
    grouped_taxonomy: ARTIFICIAL_ANALYSIS_IMAGE_GROUPS,
    global_aggregates: toAggregatedFields(globalAccumulator),
    data,
  };
}

/**
 * Fetch and enrich Artificial Analysis text-to-image leaderboard data.
 */
export async function getArtificialAnalysisImageStats(
  options: ArtificialAnalysisImageOptions = {},
): Promise<ArtificialAnalysisImageOutputPayload> {
  const apiKey = options.apiKey ?? process.env.ARTIFICIALANALYSIS_API_KEY;
  if (!apiKey) {
    return {
      fetched_at_epoch_seconds: null,
      status_code: null,
      endpoint: TEXT_TO_IMAGE_URL,
      ...enrichPayload(
        {},
        options.minModelAgeDays ?? DEFAULT_MIN_MODEL_AGE_DAYS,
      ),
    };
  }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    const response = await fetch(TEXT_TO_IMAGE_URL, {
      headers: { "x-api-key": apiKey },
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!response.ok) {
      throw new Error(
        `Artificial Analysis image request failed: ${response.status}`,
      );
    }

    const rawPayload = (await response.json()) as RawTextToImagePayload;
    return {
      fetched_at_epoch_seconds: nowEpochSeconds(),
      status_code: response.status,
      endpoint: TEXT_TO_IMAGE_URL,
      ...enrichPayload(
        rawPayload,
        options.minModelAgeDays ?? DEFAULT_MIN_MODEL_AGE_DAYS,
      ),
    };
  } catch {
    return {
      fetched_at_epoch_seconds: null,
      status_code: null,
      endpoint: TEXT_TO_IMAGE_URL,
      ...enrichPayload(
        {},
        options.minModelAgeDays ?? DEFAULT_MIN_MODEL_AGE_DAYS,
      ),
    };
  }
}
