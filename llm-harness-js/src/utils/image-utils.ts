import { readFile } from "node:fs/promises";

import sharp from "sharp";

async function loadImage(imageSource: string): Promise<Buffer> {
  if (imageSource.startsWith("http://") || imageSource.startsWith("https://")) {
    const response = await fetch(imageSource);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch image: ${response.status} ${response.statusText}`,
      );
    }
    return Buffer.from(await response.arrayBuffer());
  }

  return readFile(imageSource);
}

export async function loadImageBase64(
  imageSource: string,
  maxSize: [number, number] = [768, 768],
  format: "jpeg" | "png" | "webp" = "jpeg",
): Promise<string> {
  const imageBuffer = await loadImage(imageSource);
  const [maxWidth, maxHeight] = maxSize;

  let pipeline = sharp(imageBuffer).resize({
    width: maxWidth,
    height: maxHeight,
    fit: "inside",
    withoutEnlargement: false,
  });

  if (format === "jpeg") {
    pipeline = pipeline.jpeg();
  } else if (format === "png") {
    pipeline = pipeline.png();
  } else {
    pipeline = pipeline.webp();
  }

  const result = await pipeline.toBuffer();
  return result.toString("base64");
}
