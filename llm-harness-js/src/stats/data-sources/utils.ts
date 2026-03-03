export type NumberOrNull = number | null;

export function nowEpochSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

export function finiteNumbers(values: unknown[]): number[] {
  return values
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
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
