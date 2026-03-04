import { fetchWithTimeout, nowEpochSeconds } from "../utils.js";

const DEFAULT_SCRAPE_URL = "https://artificialanalysis.ai/leaderboards/models";
const DEFAULT_TIMEOUT_MS = 30_000;
const ROW_DETECTION_KEY = "intelligence_index";
const SPARSE_COLUMN_NULL_RATIO = 0.5;
const MODEL_SEARCH_BACKTRACK_CHARS = 20_000;
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
const EVALUATION_EXPLICIT_KEYS = new Set([
  "intelligence_index_cost",
  "intelligence_index_per_m_output_tokens",
]);

function pickEvaluations(row: JsonObject): JsonObject {
  const evaluations: JsonObject = {};
  for (const [key, value] of Object.entries(row)) {
    if (EVALUATION_EXPLICIT_KEYS.has(key)) {
      if (key === "intelligence_index_cost") {
        evaluations[key] = asRecord(value);
      } else {
        evaluations[key] = value;
      }
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

  delete evaluations.omniscience;
  delete evaluations.omniscience_accuracy;
  delete evaluations.omniscience_hallucination_rate;
  delete evaluations.intelligence_index_is_estimated;
  delete evaluations.intelligence_index;
  delete evaluations.agentic_index;
  delete evaluations.coding_index;
  delete evaluations.intelligence_index_per_m_output_tokens;
  delete evaluations.intelligence_index_cost;
  return evaluations;
}

function pickIntelligence(row: JsonObject): JsonObject {
  const intelligence: JsonObject = {};
  if (typeof row.intelligence_index === "number") {
    intelligence.intelligence_index = row.intelligence_index;
  }
  if (typeof row.agentic_index === "number") {
    intelligence.agentic_index = row.agentic_index;
  }
  if (typeof row.coding_index === "number") {
    intelligence.coding_index = row.coding_index;
  }
  if (typeof row.omniscience === "number") {
    intelligence.omniscience_index = row.omniscience;
  }
  const omniscienceBreakdown = asRecord(row.omniscience_breakdown);
  const omniscienceTotal = asRecord(omniscienceBreakdown.total);
  if (typeof omniscienceTotal.accuracy === "number") {
    intelligence.omniscience_accuracy = omniscienceTotal.accuracy;
  }
  if (typeof omniscienceTotal.hallucination_rate === "number") {
    intelligence.omniscience_hallucination_rate =
      omniscienceTotal.hallucination_rate;
  }
  const intelligenceIndexCost = asRecord(row.intelligence_index_cost);
  if (typeof intelligenceIndexCost.total_cost === "number") {
    intelligence.intelligence_index_cost_total_cost =
      intelligenceIndexCost.total_cost;
  }
  return intelligence;
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
    const creator = asRecord(row.creator);
    const modelCreators = asRecord(row.model_creators);
    const providerName =
      typeof creator.name === "string"
        ? creator.name
        : typeof row.provider === "string"
          ? row.provider
          : null;
    const providerSlug =
      typeof providerName === "string"
        ? providerName
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")
            .replace(/^-+|-+$/g, "")
        : null;
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

    for (const column of keepSet) {
      if (column === "id") {
        selectedRow.id =
          providerSlug && modelSlug
            ? `${providerSlug}/${modelSlug}`
            : (modelSlug ?? row.id ?? null);
        continue;
      }
      if (column === "model_url") {
        selectedRow.model_url =
          row.model_url ?? (typeof row.id === "string" ? row.id : null);
        continue;
      }
      if (column === "model_id") {
        selectedRow.model_id =
          creatorSlug && modelUrlSlug
            ? `${creatorSlug}/${modelUrlSlug}`
            : (modelUrlSlug ?? row.model_url ?? null);
        continue;
      }
      if (column === "name") {
        selectedRow.name =
          row.short_name ??
          row.shortName ??
          row.name ??
          (typeof row.slug === "string" ? row.slug : null);
        continue;
      }
      if (column === "provider") {
        selectedRow.provider =
          providerSlug ??
          creator.name ??
          modelCreators.name ??
          row.model_creator_id ??
          row.creator_name ??
          null;
        continue;
      }
      if (column === "logo") {
        selectedRow.logo = toAbsoluteAaLogoUrl(
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
        continue;
      }
      if (column === "attachment") {
        selectedRow.attachment =
          Boolean(row.input_modality_image) ||
          Boolean(row.input_modality_video) ||
          Boolean(row.input_modality_speech);
        continue;
      }
      if (column === "reasoning") {
        selectedRow.reasoning =
          typeof row.reasoning_model === "boolean"
            ? row.reasoning_model
            : typeof row.isReasoning === "boolean"
              ? row.isReasoning
              : null;
        continue;
      }
      if (column === "reasoning_model") {
        selectedRow.reasoning_model =
          typeof row.reasoning_model === "boolean"
            ? row.reasoning_model
            : typeof row.isReasoning === "boolean"
              ? row.isReasoning
              : null;
        continue;
      }
      if (column === "input_modalities") {
        const inputModalities = [
          row.input_modality_text ? "text" : null,
          row.input_modality_image ? "image" : null,
          row.input_modality_video ? "video" : null,
          row.input_modality_speech ? "speech" : null,
        ].filter((value): value is string => value != null);
        selectedRow.input_modalities = inputModalities;
        continue;
      }
      if (column === "output_modalities") {
        const outputModalities = [
          row.output_modality_text ? "text" : null,
          row.output_modality_image ? "image" : null,
          row.output_modality_video ? "video" : null,
          row.output_modality_speech ? "speech" : null,
        ].filter((value): value is string => value != null);
        selectedRow.output_modalities = outputModalities;
        continue;
      }
      if (column === "release_date") {
        selectedRow.release_date =
          typeof row.release_date === "string" ? row.release_date : null;
        continue;
      }
      if (column === "input_tokens") {
        const intelligenceTokenCounts = asRecord(
          row.intelligence_index_token_counts,
        );
        selectedRow.input_tokens =
          row.input_tokens ??
          intelligenceTokenCounts.input_tokens ??
          row.total_input_tokens_api ??
          null;
        continue;
      }
      if (column === "output_tokens") {
        const intelligenceTokenCounts = asRecord(
          row.intelligence_index_token_counts,
        );
        selectedRow.output_tokens =
          row.output_tokens ??
          intelligenceTokenCounts.output_tokens ??
          row.total_answer_tokens_api ??
          null;
        continue;
      }
      if (column === "median_speed") {
        selectedRow.median_speed =
          row.median_output_speed ??
          asRecord(row.timescaleData).median_output_speed ??
          null;
        continue;
      }
      if (column === "median_time") {
        selectedRow.median_time =
          row.median_time_to_first_chunk ??
          asRecord(row.timescaleData).median_time_to_first_chunk ??
          null;
        continue;
      }
      if (column === "evaluations") {
        selectedRow.evaluations = pickEvaluations(row);
        continue;
      }
      if (column === "intelligence") {
        selectedRow.intelligence = pickIntelligence(row);
        continue;
      }
      selectedRow[column] = row[column] ?? null;
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
