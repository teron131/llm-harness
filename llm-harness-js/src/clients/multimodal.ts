import { readFile } from "node:fs/promises";
import { PathLike } from "node:fs";
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
  public readonly role = "user";
  public readonly content: Array<Record<string, unknown>>;

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
      if (labelPages) {
        this.content.push(createTextBlock(`Page ${idx + 1}:`));
      }

      if (item instanceof Uint8Array) {
        this.content.push(...this.fromBytes(item, mimeType));
      } else {
        this.content.push(...this.fromPath(item));
      }
    });

    if (description) {
      this.content.push(createTextBlock(description));
    }
  }

  private fromBytes(
    data: Uint8Array,
    mimeType: string,
  ): Array<Record<string, unknown>> {
    const dataUrl = `data:${mimeType};base64,${encodeBase64(data)}`;
    return [createImageBlock(dataUrl)];
  }

  private fromPath(
    filePath: string | PathLike,
  ): Array<Record<string, unknown>> {
    const normalizedPath = String(filePath);
    const suffix = path.extname(normalizedPath).toLowerCase();
    const supported = SUPPORTED_EXTENSIONS[suffix];

    if (!supported) {
      throw new Error(
        `Unsupported extension: ${suffix}. Supported: ${Object.keys(SUPPORTED_EXTENSIONS).sort().join(", ")}`,
      );
    }

    const [category, mimeType] = supported;

    if (category === "text") {
      throw new Error(
        "Text file support requires async loader; use MediaMessage.fromPathAsync for text media",
      );
    }

    return [
      {
        __deferred_path__: normalizedPath,
        __category__: category,
        __mime_type__: mimeType,
      },
    ];
  }

  static async fromPathAsync(options: {
    paths?: string | Uint8Array | Array<string | Uint8Array>;
    media?: string | Uint8Array | Array<string | Uint8Array>;
    description?: string;
    labelPages?: boolean;
    mimeType?: string;
  }): Promise<MediaMessage> {
    const message = new MediaMessage(options);
    const resolvedContent: Array<Record<string, unknown>> = [];

    for (const block of message.content) {
      if (!("__deferred_path__" in block)) {
        resolvedContent.push(block);
        continue;
      }

      const filePath = String(block.__deferred_path__);
      const category = String(block.__category__) as SupportedCategory;
      const mimeType = String(block.__mime_type__);
      const bytes = await readFile(filePath);

      if (category === "text") {
        resolvedContent.push(createTextBlock(bytes.toString("utf-8")));
        continue;
      }

      const encoded = encodeBase64(bytes);
      const dataUrl = `data:${mimeType};base64,${encoded}`;

      if (category === "image" || category === "video") {
        resolvedContent.push(createImageBlock(dataUrl));
      } else if (category === "file") {
        resolvedContent.push(createFileBlock(path.basename(filePath), dataUrl));
      } else if (category === "audio") {
        resolvedContent.push(
          createAudioBlock(
            encoded,
            path.extname(filePath).toLowerCase() === ".wav" ? "wav" : "mp3",
          ),
        );
      }
    }

    message.content.length = 0;
    message.content.push(...resolvedContent);
    return message;
  }
}
