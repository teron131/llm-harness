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
};

export type ArtificialAnalysisScrapedPayload = {
  fetched_at_epoch_seconds: number | null;
  data: JsonObject[];
};

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

/**
 * Scrape expanded LLM leaderboard rows from Artificial Analysis page payload.
 *
 * This parser targets Next.js flight chunks embedded in HTML and is best-effort.
 * It is failure-safe and returns an empty payload on any fetch/parse failure.
 */
export async function getArtificialAnalysisScrapedStats(
  options: ArtificialAnalysisScraperOptions = {},
): Promise<ArtificialAnalysisScrapedPayload> {
  try {
    const url = options.url ?? DEFAULT_SCRAPE_URL;
    const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const shouldFlatten = options.flatten ?? true;
    const shouldDropMostlyNullColumns = options.dropMostlyNullColumns ?? true;

    const response = await fetchWithTimeout(url, {}, timeoutMs);
    if (!response.ok) {
      throw new Error(`Artificial Analysis scrape failed: ${response.status}`);
    }
    const pageHtml = await response.text();
    const corpus = extractFlightCorpus(pageHtml);
    const rawRows = extractRowsFromCorpus(corpus);
    const normalizedRows = shouldFlatten
      ? rawRows.map(flattenExpandedRow)
      : rawRows;
    const data = shouldDropMostlyNullColumns
      ? dropMostlyNullColumns(normalizedRows, SPARSE_COLUMN_NULL_RATIO)
      : normalizedRows;

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
