const MODELS_DEV_URL = "https://models.dev/api.json";
const LOOKBACK_DAYS = 365;
const REQUEST_TIMEOUT_MS = 30_000;

type NumberOrNull = number | null;

export type ModelRecord = {
  id?: string;
  name?: string;
  family?: string;
  release_date?: string;
  last_updated?: string;
  open_weights?: boolean;
  reasoning?: boolean;
  tool_call?: boolean;
  cost?: {
    input?: number;
    output?: number;
    cache_read?: number;
    cache_write?: number;
    output_audio?: number;
  };
  limit?: {
    context?: number;
    output?: number;
  };
  modalities?: {
    input?: string[];
    output?: string[];
  };
  [key: string]: unknown;
};

export type ProviderRecord = {
  id?: string;
  name?: string;
  api?: string;
  models?: Record<string, ModelRecord>;
  [key: string]: unknown;
};

export type ModelsDevPayload = Record<string, ProviderRecord>;

export type ModelsDevFlatModel = {
  provider_id: string;
  provider_name: string;
  model_id: string;
  model: ModelRecord;
};

type ModelsDevSourcePayload = {
  fetched_at_epoch_seconds: number | null;
  status_code: number | null;
  payload: ModelsDevPayload;
};

export type ModelsDevOutputPayload = {
  fetched_at_epoch_seconds: number | null;
  status_code: number | null;
  models: ModelsDevFlatModel[];
};

export type ModelsDevOptions = Record<string, never>;

function nowEpochSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

function isoDateDaysAgo(days: number): string {
  return new Date(Date.now() - days * 24 * 60 * 60 * 1000)
    .toISOString()
    .slice(0, 10);
}

function isRecentDate(
  isoDate: string | undefined,
  cutoffIsoDate: string,
): boolean {
  if (!isoDate) {
    return false;
  }
  return isoDate >= cutoffIsoDate;
}

function asFinite(value: unknown): NumberOrNull {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

async function fetchModelsDev(): Promise<ModelsDevSourcePayload> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const response = await fetch(MODELS_DEV_URL, { signal: controller.signal });
  clearTimeout(timeout);

  if (!response.ok) {
    throw new Error(`models.dev request failed: ${response.status}`);
  }

  const payload = (await response.json()) as ModelsDevPayload;
  const sourcePayload: ModelsDevSourcePayload = {
    fetched_at_epoch_seconds: nowEpochSeconds(),
    status_code: response.status,
    payload,
  };
  return sourcePayload;
}

function flattenModels(payload: ModelsDevPayload): ModelsDevFlatModel[] {
  const rows: ModelsDevFlatModel[] = [];
  for (const [providerId, provider] of Object.entries(payload)) {
    const providerName = provider.name ?? providerId;
    const models = provider.models ?? {};
    for (const [modelId, model] of Object.entries(models)) {
      rows.push({
        provider_id: providerId,
        provider_name: providerName,
        model_id: model.id ?? modelId,
        model,
      });
    }
  }
  return rows;
}

function rankRecentModels(
  models: ModelsDevFlatModel[],
  cutoffIsoDate: string,
): ModelsDevFlatModel[] {
  return models
    .filter((row) => isRecentDate(row.model.release_date, cutoffIsoDate))
    .sort((left, right) => {
      const leftOutputCost =
        asFinite(left.model.cost?.output) ?? Number.POSITIVE_INFINITY;
      const rightOutputCost =
        asFinite(right.model.cost?.output) ?? Number.POSITIVE_INFINITY;
      if (leftOutputCost !== rightOutputCost) {
        return leftOutputCost - rightOutputCost;
      }
      return (right.model.release_date ?? "").localeCompare(
        left.model.release_date ?? "",
      );
    });
}

export async function getModelsDevStats(
  _options: ModelsDevOptions = {},
): Promise<ModelsDevOutputPayload> {
  try {
    const sourcePayload = await fetchModelsDev();
    const cutoffIsoDate = isoDateDaysAgo(LOOKBACK_DAYS);
    return {
      fetched_at_epoch_seconds: sourcePayload.fetched_at_epoch_seconds,
      status_code: sourcePayload.status_code,
      models: rankRecentModels(
        flattenModels(sourcePayload.payload),
        cutoffIsoDate,
      ),
    };
  } catch {
    return {
      fetched_at_epoch_seconds: null,
      status_code: null,
      models: [],
    };
  }
}
