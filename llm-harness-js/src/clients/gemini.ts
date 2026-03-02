import { GoogleGenAI } from "@google/genai";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";

const DEFAULT_MODEL = "gemini-2.5-flash-preview-09-2025";

function resolveGeminiApiKey(
  errorMessage = "GEMINI_API_KEY (or GOOGLE_API_KEY) must be set",
): string {
  const apiKey = process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    throw new Error(errorMessage);
  }
  if (!process.env.GOOGLE_API_KEY) {
    process.env.GOOGLE_API_KEY = apiKey;
  }
  return apiKey;
}

export function ChatGemini({
  model,
  temperature = 0,
  ...kwargs
}: {
  model: string;
  temperature?: number;
  [key: string]: unknown;
}) {
  resolveGeminiApiKey();
  return new ChatGoogleGenerativeAI({
    model,
    temperature,
    ...(kwargs as Record<string, unknown>),
  });
}

export function GeminiEmbeddings({
  model = "models/text-embedding-004",
  ...kwargs
}: {
  model?: string;
  [key: string]: unknown;
} = {}) {
  resolveGeminiApiKey();
  const resolvedModel = model.startsWith("models/") ? model : `models/${model}`;
  return new GoogleGenerativeAIEmbeddings({
    model: resolvedModel,
    ...(kwargs as Record<string, unknown>),
  });
}

export async function createGeminiCache({
  filePath,
  model = DEFAULT_MODEL,
  apiKey,
}: {
  filePath: string;
  model?: string;
  apiKey?: string;
}): Promise<string> {
  const resolvedApiKey = apiKey ?? resolveGeminiApiKey("API key not found");
  const ai = new GoogleGenAI({ apiKey: resolvedApiKey });

  const uploaded = await ai.files.upload({
    file: filePath,
  });

  const uploadedName = uploaded.name;
  if (!uploadedName) {
    throw new Error("Upload failed: missing file name");
  }

  let fileRef = await ai.files.get({ name: uploadedName });
  while (fileRef.state === "PROCESSING") {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    fileRef = await ai.files.get({ name: uploadedName });
  }

  if (fileRef.state === "FAILED") {
    throw new Error(`Upload failed: ${fileRef.state}`);
  }

  if (!uploaded.uri || !uploaded.mimeType) {
    throw new Error("Upload failed: missing file URI or mime type");
  }

  const cache = await ai.caches.create({
    model,
    config: {
      contents: [
        {
          fileData: {
            fileUri: uploaded.uri,
            mimeType: uploaded.mimeType,
          },
        },
      ],
    },
  });

  if (!cache.name) {
    throw new Error("Cache creation failed: missing cache name");
  }

  return cache.name;
}
