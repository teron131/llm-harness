import { fetchWithTimeout, nowEpochSeconds } from "../utils.js";

const DEFAULT_SCRAPE_URL = "https://artificialanalysis.ai/leaderboards/models";
const DEFAULT_TIMEOUT_MS = 30_000;
const ROW_DETECTION_KEY = "intelligence_index";

type JsonObject = Record<string, unknown>;

export type ArtificialAnalysisScraperOptions = {
  url?: string;
  timeoutMs?: number;
  flatten?: boolean;
};

export type ArtificialAnalysisScrapedPayload = {
  fetched_at_epoch_seconds: number | null;
  status_code: number | null;
  total_rows: number;
  total_columns: number;
  columns: string[];
  rows: JsonObject[];
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
  const matches = [
    ...pageHtml.matchAll(
      /self\.__next_f\.push\(\[1,\"([\s\S]*?)\"\]\)<\/script>/g,
    ),
  ];
  return matches.map((match) => decodeFlightChunk(match[1] ?? "")).join("\n");
}

function findObjectEnd(corpus: string, startIndex: number): number {
  let depth = 0;
  let inString = false;
  let escaping = false;

  for (let index = startIndex; index < corpus.length; index += 1) {
    const char = corpus[index];
    if (!char) {
      continue;
    }
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

function flattenExpandedRow(row: JsonObject): JsonObject {
  const timescaleData = asRecord(row.timescaleData);
  const responseTimeMetrics = asRecord(row.end_to_end_response_time_metrics);
  const firstPerformanceRow = Array.isArray(row.performanceByPromptLength)
    ? asRecord(row.performanceByPromptLength[0])
    : {};

  return {
    ...row,
    median_output_speed:
      row.median_output_speed ?? timescaleData.median_output_speed ?? null,
    percentile_05_output_speed:
      row.percentile_05_output_speed ??
      timescaleData.percentile_05_output_speed ??
      null,
    quartile_25_output_speed:
      row.quartile_25_output_speed ??
      timescaleData.quartile_25_output_speed ??
      null,
    quartile_75_output_speed:
      row.quartile_75_output_speed ??
      timescaleData.quartile_75_output_speed ??
      null,
    percentile_95_output_speed:
      row.percentile_95_output_speed ??
      timescaleData.percentile_95_output_speed ??
      null,
    median_time_to_first_chunk:
      row.median_time_to_first_chunk ??
      timescaleData.median_time_to_first_chunk ??
      null,
    median_estimated_total_seconds_for_100_output_tokens:
      row.median_estimated_total_seconds_for_100_output_tokens ??
      timescaleData.median_estimated_total_seconds_for_100_output_tokens ??
      null,
    p05_total_time:
      row.p05_total_time ?? responseTimeMetrics.p05_total_time ?? null,
    p25_total_time:
      row.p25_total_time ?? responseTimeMetrics.p25_total_time ?? null,
    p75_total_time:
      row.p75_total_time ?? responseTimeMetrics.p75_total_time ?? null,
    p95_total_time:
      row.p95_total_time ?? responseTimeMetrics.p95_total_time ?? null,
    prompt_length_type_default:
      row.prompt_length_type_default ??
      firstPerformanceRow.prompt_length_type ??
      null,
  };
}

function extractRowsFromCorpus(corpus: string): JsonObject[] {
  const detectionToken = `"${ROW_DETECTION_KEY}":`;
  const hitIndices: number[] = [];
  let cursor = 0;
  while (true) {
    const index = corpus.indexOf(detectionToken, cursor);
    if (index === -1) {
      break;
    }
    hitIndices.push(index);
    cursor = index + detectionToken.length;
  }

  const rowsById = new Map<string, JsonObject>();
  for (const hitIndex of hitIndices) {
    const searchStart = Math.max(0, hitIndex - 20_000);
    for (let backIndex = hitIndex; backIndex >= searchStart; backIndex -= 1) {
      if (corpus[backIndex] !== "{") {
        continue;
      }
      const endIndex = findObjectEnd(corpus, backIndex);
      if (endIndex === -1 || endIndex < hitIndex) {
        continue;
      }
      const candidateText = corpus.slice(backIndex, endIndex + 1);
      let candidate: unknown;
      try {
        candidate = JSON.parse(candidateText);
      } catch {
        continue;
      }
      const row = asRecord(candidate);
      if (!(ROW_DETECTION_KEY in row)) {
        continue;
      }
      const rowId =
        typeof row.id === "string"
          ? row.id
          : typeof row.model_id === "string"
            ? row.model_id
            : typeof row.slug === "string"
              ? row.slug
              : null;
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

    const response = await fetchWithTimeout(url, {}, timeoutMs);
    if (!response.ok) {
      throw new Error(`Artificial Analysis scrape failed: ${response.status}`);
    }
    const pageHtml = await response.text();
    const corpus = extractFlightCorpus(pageHtml);
    const rawRows = extractRowsFromCorpus(corpus);
    const rows = shouldFlatten ? rawRows.map(flattenExpandedRow) : rawRows;
    const columns = [
      ...new Set(rows.flatMap((row) => Object.keys(row))),
    ].sort();

    return {
      fetched_at_epoch_seconds: nowEpochSeconds(),
      status_code: response.status,
      total_rows: rows.length,
      total_columns: columns.length,
      columns,
      rows,
    };
  } catch {
    return {
      fetched_at_epoch_seconds: null,
      status_code: null,
      total_rows: 0,
      total_columns: 0,
      columns: [],
      rows: [],
    };
  }
}
