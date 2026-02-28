import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { getMatchModelsUnion } from "./data-sources/matcher.js";

const OUTPUT_PATH = resolve(".cache/eval_models_union_selected.json");
const CACHE_DIR = resolve(".cache");

type JsonObject = Record<string, unknown>;

export type ModelStatsSelectedModel = {
  id: string | null;
  name: string | null;
  provider: string | null;
  logo: string;
  attachment: boolean | null;
  reasoning: boolean | null;
  release_date: string | null;
  modalities: unknown;
  open_weights: boolean | null;
  cost: unknown;
  context_window: unknown;
  speed: JsonObject;
  evaluations: unknown;
  scores: unknown;
  percentiles: unknown;
};

export type ModelStatsSelectedPayload = {
  models: ModelStatsSelectedModel[];
};

function providerFromId(modelId: unknown): string | null {
  if (typeof modelId !== "string") {
    return null;
  }
  const slashIndex = modelId.indexOf("/");
  if (slashIndex <= 0) {
    return null;
  }
  return modelId.slice(0, slashIndex);
}

function asRecord(value: unknown): JsonObject {
  return value != null && typeof value === "object"
    ? (value as JsonObject)
    : {};
}

function buildLogo(model: JsonObject, provider: string | null): string {
  const modelCreator = asRecord(model.model_creator);
  const logoSlug = modelCreator.slug;
  if (typeof logoSlug === "string" && logoSlug.length > 0) {
    return `https://artificialanalysis.ai/img/logos/${logoSlug}_small.svg`;
  }
  return `https://models.dev/logos/${provider ?? "unknown"}.svg`;
}

function buildSpeed(model: JsonObject): JsonObject {
  return Object.fromEntries(
    Object.entries(model).filter(([key]) => key.startsWith("median_")),
  );
}

async function writeJson(path: string, payload: unknown): Promise<void> {
  await mkdir(CACHE_DIR, { recursive: true });
  await writeFile(path, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
}

export async function getModelStatsSelected(): Promise<ModelStatsSelectedPayload> {
  const matchUnion = await getMatchModelsUnion();
  const models = matchUnion.models.map(
    (unionModel): ModelStatsSelectedModel => {
      const model = asRecord(unionModel);
      const provider = providerFromId(model.id);
      return {
        id: typeof model.id === "string" ? model.id : null,
        name: typeof model.name === "string" ? model.name : null,
        provider,
        logo: buildLogo(model, provider),
        attachment:
          typeof model.attachment === "boolean" ? model.attachment : null,
        reasoning:
          typeof model.reasoning === "boolean" ? model.reasoning : null,
        release_date:
          typeof model.release_date === "string" ? model.release_date : null,
        modalities: model.modalities ?? null,
        open_weights:
          typeof model.open_weights === "boolean" ? model.open_weights : null,
        cost: model.cost ?? null,
        context_window: model.limit ?? null,
        speed: buildSpeed(model),
        evaluations: model.evaluations ?? null,
        scores: model.scores ?? null,
        percentiles: model.percentiles ?? null,
      };
    },
  );

  const outputPayload: ModelStatsSelectedPayload = { models };
  await writeJson(OUTPUT_PATH, outputPayload);
  return outputPayload;
}
