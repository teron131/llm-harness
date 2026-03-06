import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

export type JsonObject = Record<string, unknown>;
export type NumberOrNull = number | null;

export function nowEpochSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

export function asRecord(value: unknown): JsonObject {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
}

export function asFiniteNumber(value: unknown): NumberOrNull {
  if (value == null) {
    return null;
  }
  if (typeof value === "string" && value.trim().length === 0) {
    return null;
  }
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : null;
}

export function finiteNumbers(values: unknown[]): number[] {
  return values
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
}

export function meanOrNull(values: unknown[]): NumberOrNull {
  const finiteValues = finiteNumbers(values);
  if (finiteValues.length === 0) {
    return null;
  }
  const mean =
    finiteValues.reduce((sum, value) => sum + value, 0) / finiteValues.length;
  return Number(mean.toFixed(4));
}

export function percentileRank(
  values: unknown[],
  value: unknown,
): NumberOrNull {
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
  const rawPercentile = (lessOrEqualCount / finiteValues.length) * 100;
  return Number(rawPercentile.toFixed(4));
}

export async function fetchWithTimeout(
  input: string | URL,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, {
      ...init,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeout);
  }
}

export function isFreshEpochSeconds(
  fetchedAtEpochSeconds: unknown,
  ttlSeconds: number,
): boolean {
  if (typeof fetchedAtEpochSeconds !== "number") {
    return false;
  }
  const ageSeconds = nowEpochSeconds() - fetchedAtEpochSeconds;
  return ageSeconds >= 0 && ageSeconds <= ttlSeconds;
}

export async function writeJsonFile(
  path: string,
  payload: unknown,
): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
}
