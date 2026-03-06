import { asFiniteNumber, asRecord, type JsonObject } from "../utils";

export { asFiniteNumber, asRecord, type JsonObject };

export const PRIMARY_PROVIDER_ID = "openrouter" as const;
export const FALLBACK_PROVIDER_IDS = new Set(["openai", "google", "anthropic"]);

export function normalizeModelToken(value: string): string {
  return value
    .toLowerCase()
    .replace(/[._:\s]+/g, "-")
    .replace(/[^a-z0-9/-]+/g, "")
    .replace(/-+/g, "-")
    .replace(/^[-/]+|[-/]+$/g, "");
}

export function modelSlugFromModelId(modelId: unknown): string | null {
  if (typeof modelId !== "string" || modelId.length === 0) {
    return null;
  }
  const slug = modelId.split("/").at(-1);
  return slug && slug.length > 0 ? slug : null;
}

export function normalizeProviderModelId(modelId: string): string {
  const slashIndex = modelId.indexOf("/");
  if (slashIndex <= 0) {
    return modelId.toLowerCase().replace(/\./g, "-").replace(/-+/g, "-");
  }
  const provider = modelId.slice(0, slashIndex).toLowerCase();
  const baseModelId = modelId
    .slice(slashIndex + 1)
    .toLowerCase()
    .replace(/\./g, "-")
    .replace(/-+/g, "-");
  return `${provider}/${baseModelId}`;
}
