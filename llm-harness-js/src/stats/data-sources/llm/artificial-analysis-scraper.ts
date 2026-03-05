import { fetchWithTimeout, nowEpochSeconds } from "../utils.js";

const DEFAULT_SCRAPE_URL = "https://artificialanalysis.ai/leaderboards/models";
const DEFAULT_TIMEOUT_MS = 30_000;
const ROW_DETECTION_KEY = "intelligence_index";
const SPARSE_COLUMN_NULL_RATIO = 0.5;
const MODEL_SEARCH_BACKTRACK_CHARS = 20_000;
const MIN_INTELLIGENCE_COST_TOKEN_THRESHOLD = 1_000_000;
const NEXT_FLIGHT_CHUNK_REGEX =
  /self\.__next_f\.push\(\[1,\"([\s\S]*?)\"\]\)<\/script>/g;

type JsonObject = Record<string, unknown>;

export type ArtificialAnalysisScraperOptions = {
  url?: string;
  timeoutMs?: number;
  flatten?: boolean;
  dropMostlyNullColumns?: boolean;
  selectedColumns?: string[];
};

export type ArtificialAnalysisScraperProcessOptions = {
  flatten?: boolean;
  dropMostlyNullColumns?: boolean;
  selectedColumns?: string[];
};

export type ArtificialAnalysisScrapedRawPayload = {
  fetched_at_epoch_seconds: number | null;
  data: JsonObject[];
};

export type ArtificialAnalysisScrapedPayload =
  ArtificialAnalysisScrapedRawPayload;

export const ARTIFICIAL_ANALYSIS_EVALS_ONLY_COLUMNS = [
  "model_id",
  "logo",
  "intelligence",
  "intelligence_index_cost",
  "evaluations",
] as const;

function asRecord(value: unknown): JsonObject {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
}

function decodeFlightChunk(raw: string): string {
  try {
    return JSON.parse(`"${raw}"`) as string;
  } catch {
    return raw;
  }
}

function toAbsoluteAaLogoUrl(value: unknown): string | null {
  if (typeof value !== "string" || value.length === 0) {
    return null;
  }
  if (value.startsWith("http://") || value.startsWith("https://")) {
    return value;
  }
  const normalized = value.startsWith("/") ? value : `/${value}`;
  return `https://artificialanalysis.ai${normalized}`;
}

const EVALUATION_KEY_HINT_REGEX =
  /(index|bench|mmlu|gpqa|hle|aime|math|vision|omniscience|ifbench|gdpval|lcr|arc|musr|humanity)/i;
const NON_EVALUATION_KEY_REGEX =
  /(token|time|speed|price|cost|window|modality|reasoning_model|release_date|display_order|deprecated|deleted|commercial_allowed|frontier_model|is_open_weights|logo|url|license|creator|host|slug|name|id$|^id$|model_|timescale|response|performance|voice|image|audio|video|text)/i;
const EVALUATION_EXCLUDED_KEYS = new Set([
  "omniscience",
  "omniscience_accuracy",
  "omniscience_hallucination_rate",
  "intelligence_index_is_estimated",
  "intelligence_index",
  "agentic_index",
  "coding_index",
  "intelligence_index_per_m_output_tokens",
  "intelligence_index_cost",
]);
const NO_COLUMN_VALUE = Symbol("no_column_value");

function pickEvaluations(row: JsonObject): JsonObject {
  const evaluations: JsonObject = {};
  for (const [key, value] of Object.entries(row)) {
    if (EVALUATION_EXCLUDED_KEYS.has(key)) {
      continue;
    }
    if (!EVALUATION_KEY_HINT_REGEX.test(key)) {
      continue;
    }
    if (NON_EVALUATION_KEY_REGEX.test(key)) {
      continue;
    }
    if (typeof value === "number" || typeof value === "boolean") {
      evaluations[key] = value;
    }
  }
  return evaluations;
}

function pickIntelligence(row: JsonObject): JsonObject {
  const intelligenceTokenCounts = asRecord(row.intelligence_index_token_counts);
  const inputTokens =
    typeof intelligenceTokenCounts.input_tokens === "number"
      ? intelligenceTokenCounts.input_tokens
      : typeof row.total_input_tokens_api === "number"
        ? row.total_input_tokens_api
        : typeof row.input_tokens === "number"
          ? row.input_tokens
          : null;
  const tokenCountOutputTokens =
    typeof intelligenceTokenCounts.output_tokens === "number"
      ? intelligenceTokenCounts.output_tokens
      : null;
  const tokenCountAnswerTokens =
    typeof intelligenceTokenCounts.answer_tokens === "number"
      ? intelligenceTokenCounts.answer_tokens
      : null;
  const tokenCountReasoningTokens =
    typeof intelligenceTokenCounts.reasoning_tokens === "number"
      ? intelligenceTokenCounts.reasoning_tokens
      : null;
  const tokenCountOutputFromParts =
    (tokenCountAnswerTokens ?? 0) + (tokenCountReasoningTokens ?? 0) > 0
      ? (tokenCountAnswerTokens ?? 0) + (tokenCountReasoningTokens ?? 0)
      : null;
  const outputTokens =
    tokenCountOutputTokens ??
    tokenCountOutputFromParts ??
    (typeof row.total_answer_tokens_api === "number"
      ? row.total_answer_tokens_api
      : null) ??
    (typeof row.output_tokens === "number" ? row.output_tokens : null);

  const intelligence: JsonObject = {
    intelligence_index:
      typeof row.intelligence_index === "number"
        ? row.intelligence_index
        : null,
    agentic_index:
      typeof row.agentic_index === "number" ? row.agentic_index : null,
    coding_index:
      typeof row.coding_index === "number" ? row.coding_index : null,
    omniscience_index:
      typeof row.omniscience === "number" ? row.omniscience : null,
    omniscience_accuracy: null,
    omniscience_nonhallucination_rate: null,
  };
  const omniscienceBreakdown = asRecord(row.omniscience_breakdown);
  const omniscienceTotal = asRecord(omniscienceBreakdown.total);
  if (typeof omniscienceTotal.accuracy === "number") {
    intelligence.omniscience_accuracy = omniscienceTotal.accuracy;
  }
  if (typeof omniscienceTotal.hallucination_rate === "number") {
    intelligence.omniscience_nonhallucination_rate =
      omniscienceTotal.hallucination_rate;
  }
  return intelligence;
}

function pickIntelligenceIndexCost(row: JsonObject): JsonObject {
  const intelligenceTokenCounts = asRecord(row.intelligence_index_token_counts);
  const intelligenceIndexCost = asRecord(row.intelligence_index_cost);
  const inputTokens =
    typeof intelligenceTokenCounts.input_tokens === "number"
      ? intelligenceTokenCounts.input_tokens
      : typeof row.total_input_tokens_api === "number"
        ? row.total_input_tokens_api
        : typeof row.input_tokens === "number"
          ? row.input_tokens
          : null;
  const outputTokens =
    typeof intelligenceTokenCounts.output_tokens === "number"
      ? intelligenceTokenCounts.output_tokens
      : null;
  const answerTokens =
    typeof intelligenceTokenCounts.answer_tokens === "number"
      ? intelligenceTokenCounts.answer_tokens
      : null;
  const reasoningTokens =
    typeof intelligenceTokenCounts.reasoning_tokens === "number"
      ? intelligenceTokenCounts.reasoning_tokens
      : null;
  const outputFromParts =
    (answerTokens ?? 0) + (reasoningTokens ?? 0) > 0
      ? (answerTokens ?? 0) + (reasoningTokens ?? 0)
      : null;
  const totalTokens =
    outputTokens ??
    outputFromParts ??
    (typeof row.total_answer_tokens_api === "number"
      ? row.total_answer_tokens_api
      : null) ??
    (typeof row.output_tokens === "number" ? row.output_tokens : null);

  return {
    input_cost:
      typeof intelligenceIndexCost.input_cost === "number"
        ? intelligenceIndexCost.input_cost
        : null,
    reasoning_cost:
      typeof intelligenceIndexCost.reasoning_cost === "number"
        ? intelligenceIndexCost.reasoning_cost
        : null,
    output_cost:
      typeof intelligenceIndexCost.output_cost === "number"
        ? intelligenceIndexCost.output_cost
        : null,
    total_cost:
      typeof intelligenceIndexCost.total_cost === "number"
        ? intelligenceIndexCost.total_cost
        : null,
    input_tokens: inputTokens,
    reasoning_tokens: reasoningTokens,
    answer_tokens: answerTokens,
    output_tokens: outputTokens,
    total_tokens:
      typeof totalTokens === "number" &&
      totalTokens >= MIN_INTELLIGENCE_COST_TOKEN_THRESHOLD
        ? totalTokens
        : null,
  };
}

function normalizeUndefinedToNull(value: unknown): unknown {
  if (value === undefined) {
    return null;
  }
  if (Array.isArray(value)) {
    return value.map((item) => normalizeUndefinedToNull(item));
  }
  if (value != null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, nestedValue]) => [
        key,
        normalizeUndefinedToNull(nestedValue),
      ]),
    );
  }
  return value;
}

function extractFlightCorpus(pageHtml: string): string {
  const matches = [...pageHtml.matchAll(NEXT_FLIGHT_CHUNK_REGEX)];
  return matches.map((match) => decodeFlightChunk(match[1] ?? "")).join("\n");
}

function findObjectEnd(corpus: string, startIndex: number): number {
  let depth = 0;
  let inString = false;
  let escaping = false;

  for (let index = startIndex; index < corpus.length; index += 1) {
    const char = corpus[index];
    if (inString) {
      if (escaping) {
        escaping = false;
      } else if (char === "\\") {
        escaping = true;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }
    if (char === '"') {
      inString = true;
      continue;
    }
    if (char === "{") {
      depth += 1;
      continue;
    }
    if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return index;
      }
    }
  }
  return -1;
}

function parseJsonObject(value: string): JsonObject | null {
  try {
    return asRecord(JSON.parse(value));
  } catch {
    return null;
  }
}

function getRowIdentifier(row: JsonObject): string | null {
  if (typeof row.id === "string") {
    return row.id;
  }
  if (typeof row.model_id === "string") {
    return row.model_id;
  }
  if (typeof row.slug === "string") {
    return row.slug;
  }
  return null;
}

function flattenExpandedRow(row: JsonObject): JsonObject {
  const timescaleData = asRecord(row.timescaleData);
  const responseTimeMetrics = asRecord(row.end_to_end_response_time_metrics);
  const firstPerformanceRow = Array.isArray(row.performanceByPromptLength)
    ? asRecord(row.performanceByPromptLength[0])
    : {};

  const flattenedRow: JsonObject = { ...row };

  for (const source of [timescaleData, responseTimeMetrics]) {
    for (const [key, value] of Object.entries(source)) {
      if (flattenedRow[key] == null && value !== undefined) {
        flattenedRow[key] = value;
      }
    }
  }

  if (
    flattenedRow.prompt_length_type_default == null &&
    firstPerformanceRow.prompt_length_type != null
  ) {
    flattenedRow.prompt_length_type_default =
      firstPerformanceRow.prompt_length_type;
  }

  return flattenedRow;
}

function isNullLike(value: unknown): boolean {
  return (
    value == null ||
    value === "" ||
    value === "$undefined" ||
    (Array.isArray(value) && value.length === 0)
  );
}

function dropMostlyNullColumns(
  rows: JsonObject[],
  nullRatioThreshold: number,
): JsonObject[] {
  if (rows.length === 0) {
    return rows;
  }
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const columnsToDrop = new Set<string>();

  for (const column of columns) {
    let nullLikeCount = 0;
    for (const row of rows) {
      if (isNullLike(row[column])) {
        nullLikeCount += 1;
      }
    }
    if (nullLikeCount / rows.length > nullRatioThreshold) {
      columnsToDrop.add(column);
    }
  }

  if (columnsToDrop.size === 0) {
    return rows;
  }
  return rows.map((row) =>
    Object.fromEntries(
      Object.entries(row).filter(([column]) => !columnsToDrop.has(column)),
    ),
  );
}

type RowSelectionContext = {
  creator: JsonObject;
  modelCreators: JsonObject;
  providerSlug: string | null;
  modelSlug: string | null;
  creatorSlug: string | null;
  modelUrlSlug: string | null;
};

function getProviderSlug(row: JsonObject, creator: JsonObject): string | null {
  const providerName =
    typeof creator.name === "string"
      ? creator.name
      : typeof row.provider === "string"
        ? row.provider
        : null;
  if (providerName == null) {
    return null;
  }
  return providerName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function buildRowSelectionContext(row: JsonObject): RowSelectionContext {
  const creator = asRecord(row.creator);
  const modelCreators = asRecord(row.model_creators);
  const providerSlug = getProviderSlug(row, creator);
  const modelSlug =
    typeof row.slug === "string" && row.slug.length > 0 ? row.slug : null;
  const creatorSlug =
    typeof modelCreators.slug === "string" && modelCreators.slug.length > 0
      ? modelCreators.slug
      : providerSlug;
  const modelUrlSlug =
    typeof row.model_url === "string"
      ? row.model_url.replace(/^\/models\//, "")
      : null;
  return {
    creator,
    modelCreators,
    providerSlug,
    modelSlug,
    creatorSlug,
    modelUrlSlug,
  };
}

function selectModalities(row: JsonObject, type: "input" | "output"): string[] {
  return [
    row[`${type}_modality_text`] ? "text" : null,
    row[`${type}_modality_image`] ? "image" : null,
    row[`${type}_modality_video`] ? "video" : null,
    row[`${type}_modality_speech`] ? "speech" : null,
  ].filter((value): value is string => value != null);
}

function selectReasoningFlag(row: JsonObject): boolean | null {
  if (typeof row.reasoning_model === "boolean") {
    return row.reasoning_model;
  }
  if (typeof row.isReasoning === "boolean") {
    return row.isReasoning;
  }
  return null;
}

function getSelectedColumnValue(
  column: string,
  row: JsonObject,
  context: RowSelectionContext,
): unknown {
  const {
    creator,
    modelCreators,
    providerSlug,
    modelSlug,
    creatorSlug,
    modelUrlSlug,
  } = context;

  switch (column) {
    case "id":
      return providerSlug && modelSlug
        ? `${providerSlug}/${modelSlug}`
        : (modelSlug ?? row.id ?? null);
    case "model_url":
      return row.model_url ?? (typeof row.id === "string" ? row.id : null);
    case "model_id":
      return creatorSlug && modelUrlSlug
        ? `${creatorSlug}/${modelUrlSlug}`
        : (modelUrlSlug ?? row.model_url ?? null);
    case "name":
      return (
        row.short_name ??
        row.shortName ??
        row.name ??
        (typeof row.slug === "string" ? row.slug : null)
      );
    case "provider":
      return (
        providerSlug ??
        creator.name ??
        modelCreators.name ??
        row.model_creator_id ??
        row.creator_name ??
        null
      );
    case "logo":
      return toAbsoluteAaLogoUrl(
        row.logo_small_url ??
          row.logo_url ??
          row.logoSmall ??
          row.logo_small ??
          modelCreators.logo_small_url ??
          modelCreators.logo_url ??
          modelCreators.logo_small ??
          modelCreators.logo ??
          creator.logo_small_url ??
          creator.logo_url ??
          creator.logo_small ??
          creator.logo,
      );
    case "attachment":
      return (
        Boolean(row.input_modality_image) ||
        Boolean(row.input_modality_video) ||
        Boolean(row.input_modality_speech)
      );
    case "reasoning":
    case "reasoning_model":
      return selectReasoningFlag(row);
    case "input_modalities":
      return selectModalities(row, "input");
    case "output_modalities":
      return selectModalities(row, "output");
    case "release_date":
      return typeof row.release_date === "string" ? row.release_date : null;
    case "input_tokens": {
      const intelligenceTokenCounts = asRecord(
        row.intelligence_index_token_counts,
      );
      return (
        intelligenceTokenCounts.input_tokens ??
        row.total_input_tokens_api ??
        row.input_tokens ??
        null
      );
    }
    case "output_tokens": {
      const intelligenceTokenCounts = asRecord(
        row.intelligence_index_token_counts,
      );
      const answerTokens =
        typeof intelligenceTokenCounts.answer_tokens === "number"
          ? intelligenceTokenCounts.answer_tokens
          : null;
      const reasoningTokens =
        typeof intelligenceTokenCounts.reasoning_tokens === "number"
          ? intelligenceTokenCounts.reasoning_tokens
          : null;
      const outputFromParts =
        (answerTokens ?? 0) + (reasoningTokens ?? 0) > 0
          ? (answerTokens ?? 0) + (reasoningTokens ?? 0)
          : null;
      return (
        intelligenceTokenCounts.output_tokens ??
        outputFromParts ??
        row.total_answer_tokens_api ??
        row.output_tokens ??
        null
      );
    }
    case "median_speed":
      return (
        row.median_output_speed ??
        asRecord(row.timescaleData).median_output_speed ??
        null
      );
    case "median_time":
      return (
        row.median_time_to_first_chunk ??
        asRecord(row.timescaleData).median_time_to_first_chunk ??
        null
      );
    case "evaluations":
      return pickEvaluations(row);
    case "intelligence":
      return pickIntelligence(row);
    case "intelligence_index_cost":
      return pickIntelligenceIndexCost(row);
    default:
      return NO_COLUMN_VALUE;
  }
}

function selectColumns(
  rows: JsonObject[],
  selectedColumns: string[],
): JsonObject[] {
  const keepSet = new Set(
    selectedColumns.filter(
      (column) => typeof column === "string" && column.length > 0,
    ),
  );
  if (keepSet.size === 0) {
    return rows;
  }
  return rows.map((row) => {
    const selectedRow: JsonObject = {};
    const context = buildRowSelectionContext(row);

    for (const column of keepSet) {
      const columnValue = getSelectedColumnValue(column, row, context);
      if (columnValue !== NO_COLUMN_VALUE) {
        selectedRow[column] = normalizeUndefinedToNull(columnValue);
      } else {
        selectedRow[column] = normalizeUndefinedToNull(row[column] ?? null);
      }
    }
    return selectedRow;
  });
}

function extractRowsFromCorpus(corpus: string): JsonObject[] {
  const detectionToken = `"${ROW_DETECTION_KEY}":`;
  const rowsById = new Map<string, JsonObject>();

  let cursor = 0;
  while (true) {
    const hitIndex = corpus.indexOf(detectionToken, cursor);
    if (hitIndex === -1) {
      break;
    }
    cursor = hitIndex + detectionToken.length;

    const searchStart = Math.max(0, hitIndex - MODEL_SEARCH_BACKTRACK_CHARS);
    for (let backIndex = hitIndex; backIndex >= searchStart; backIndex -= 1) {
      if (corpus[backIndex] !== "{") {
        continue;
      }
      const endIndex = findObjectEnd(corpus, backIndex);
      if (endIndex === -1 || endIndex < hitIndex) {
        continue;
      }
      const candidateText = corpus.slice(backIndex, endIndex + 1);
      const row = parseJsonObject(candidateText);
      if (!row) {
        continue;
      }
      if (!(ROW_DETECTION_KEY in row)) {
        continue;
      }
      const rowId = getRowIdentifier(row);
      if (!rowId) {
        continue;
      }
      rowsById.set(rowId, row);
      break;
    }
  }
  return [...rowsById.values()];
}

export function processArtificialAnalysisScrapedRows(
  rows: JsonObject[],
  options: ArtificialAnalysisScraperProcessOptions = {},
): JsonObject[] {
  const shouldFlatten = options.flatten ?? true;
  const shouldDropMostlyNullColumns = options.dropMostlyNullColumns ?? true;
  const selectedColumns = options.selectedColumns ?? [];

  const normalizedRows = shouldFlatten ? rows.map(flattenExpandedRow) : rows;
  const cleanedRows = shouldDropMostlyNullColumns
    ? dropMostlyNullColumns(normalizedRows, SPARSE_COLUMN_NULL_RATIO)
    : normalizedRows;
  return selectColumns(cleanedRows, selectedColumns);
}

/**
 * Fetch raw rows from Artificial Analysis leaderboard page payload.
 *
 * This function intentionally performs no flattening/cleaning/selection.
 */
export async function getArtificialAnalysisScrapedRawStats(
  options: Pick<ArtificialAnalysisScraperOptions, "url" | "timeoutMs"> = {},
): Promise<ArtificialAnalysisScrapedRawPayload> {
  try {
    const url = options.url ?? DEFAULT_SCRAPE_URL;
    const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;

    const response = await fetchWithTimeout(url, {}, timeoutMs);
    if (!response.ok) {
      throw new Error(`Artificial Analysis scrape failed: ${response.status}`);
    }
    const pageHtml = await response.text();
    const corpus = extractFlightCorpus(pageHtml);
    const data = extractRowsFromCorpus(corpus);

    return {
      fetched_at_epoch_seconds: nowEpochSeconds(),
      data,
    };
  } catch {
    return {
      fetched_at_epoch_seconds: null,
      data: [],
    };
  }
}

/**
 * Scrape expanded LLM leaderboard rows from Artificial Analysis page payload.
 *
 * This parser targets Next.js flight chunks embedded in HTML and is best-effort.
 * It is failure-safe and returns an empty payload on any fetch/parse failure.
 */
export async function getArtificialAnalysisScrapedStats(
  options: ArtificialAnalysisScraperOptions = {},
): Promise<ArtificialAnalysisScrapedPayload> {
  const rawPayload = await getArtificialAnalysisScrapedRawStats(options);
  return {
    fetched_at_epoch_seconds: rawPayload.fetched_at_epoch_seconds,
    data: processArtificialAnalysisScrapedRows(rawPayload.data, options),
  };
}

export async function getArtificialAnalysisScrapedEvalsOnlyStats(
  options: Omit<ArtificialAnalysisScraperOptions, "selectedColumns"> = {},
): Promise<ArtificialAnalysisScrapedPayload> {
  return getArtificialAnalysisScrapedStats({
    ...options,
    selectedColumns: [...ARTIFICIAL_ANALYSIS_EVALS_ONLY_COLUMNS],
  });
}
