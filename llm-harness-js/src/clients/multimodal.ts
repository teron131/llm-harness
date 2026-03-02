import { readFileSync } from "node:fs";
import path from "node:path";

type SupportedCategory = "image" | "video" | "audio" | "file" | "text";

const SUPPORTED_EXTENSIONS: Record<string, [SupportedCategory, string]> = {
  ".jpg": ["image", "image/jpeg"],
  ".jpeg": ["image", "image/jpeg"],
  ".png": ["image", "image/png"],
  ".gif": ["image", "image/gif"],
  ".webp": ["image", "image/webp"],
  ".mp4": ["video", "video/mp4"],
  ".mpeg": ["video", "video/mpeg"],
  ".mov": ["video", "video/quicktime"],
  ".webm": ["video", "video/webm"],
  ".mp3": ["audio", "audio/mpeg"],
  ".wav": ["audio", "audio/wav"],
  ".pdf": ["file", "application/pdf"],
  ".txt": ["text", "text/plain"],
  ".md": ["text", "text/markdown"],
};

function encodeBase64(data: Uint8Array): string {
  return Buffer.from(data).toString("base64");
}

function createTextBlock(text: string): Record<string, unknown> {
  return { type: "text", text };
}

function createImageBlock(dataUrl: string): Record<string, unknown> {
  return { type: "image_url", image_url: { url: dataUrl } };
}

function createFileBlock(
  filename: string,
  dataUrl: string,
): Record<string, unknown> {
  return { type: "file", file: { filename, file_data: dataUrl } };
}

function createAudioBlock(
  encodedData: string,
  format: "wav" | "mp3",
): Record<string, unknown> {
  return { type: "input_audio", input_audio: { data: encodedData, format } };
}

export class MediaMessage {
  readonly role = "user";
  readonly content: Record<string, unknown>[];

  constructor({
    paths,
    media,
    description = "",
    labelPages = false,
    mimeType = "image/jpeg",
  }: {
    paths?: string | Uint8Array | Array<string | Uint8Array>;
    media?: string | Uint8Array | Array<string | Uint8Array>;
    description?: string;
    labelPages?: boolean;
    mimeType?: string;
  }) {
    const mediaInput = paths ?? media;
    if (!mediaInput) {
      throw new Error("Either 'paths' or 'media' must be provided");
    }

    const items = Array.isArray(mediaInput) ? mediaInput : [mediaInput];
    this.content = [];

    items.forEach((item, idx) => {
      const blocks =
        item instanceof Uint8Array
          ? this.fromBytes(item, mimeType)
          : this.fromPath(item);

      if (labelPages && blocks.length > 0) {
        this.content.push(createTextBlock(`Page ${idx + 1}:`));
      }

      this.content.push(...blocks);
    });

    if (description) {
      this.content.push(createTextBlock(description));
    }
  }

  private fromBytes(
    data: Uint8Array,
    mimeType: string,
  ): Record<string, unknown>[] {
    const dataUrl = `data:${mimeType};base64,${encodeBase64(data)}`;
    return [createImageBlock(dataUrl)];
  }

  private fromPath(filePath: string): Record<string, unknown>[] {
    const suffix = path.extname(filePath).toLowerCase();
    const supported = SUPPORTED_EXTENSIONS[suffix];

    if (!supported) {
      throw new Error(
        `Unsupported extension: ${suffix}. Supported: ${Object.keys(SUPPORTED_EXTENSIONS).sort().join(", ")}`,
      );
    }

    const [category, mimeType] = supported;

    if (category === "text") {
      return [createTextBlock(readFileSync(filePath, "utf-8"))];
    }

    const bytes = readFileSync(filePath);
    const encoded = encodeBase64(bytes);
    const dataUrl = `data:${mimeType};base64,${encoded}`;

    if (category === "image" || category === "video") {
      return [createImageBlock(dataUrl)];
    }

    if (category === "file") {
      return [createFileBlock(path.basename(filePath), dataUrl)];
    }

    if (category === "audio") {
      return [createAudioBlock(encoded, suffix === ".wav" ? "wav" : "mp3")];
    }

    return [];
  }

  static fromPathAsync(options: {
    paths?: string | Uint8Array | Array<string | Uint8Array>;
    media?: string | Uint8Array | Array<string | Uint8Array>;
    description?: string;
    labelPages?: boolean;
    mimeType?: string;
  }): Promise<MediaMessage> {
    return Promise.resolve(new MediaMessage(options));
  }
}
