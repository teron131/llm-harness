const YOUTUBE_PATTERNS = [
  /youtube\.com\/watch\?v=/,
  /youtu\.be\//,
  /youtube\.com\/embed\//,
  /youtube\.com\/v\//,
];

export function cleanText(text: string): string {
  return text
    .replace(/\n{3,}/g, "\n\n")
    .replace(/ {2,}/g, " ")
    .trim();
}

export function cleanYoutubeUrl(url: string): string {
  if (url.includes("youtube.com/watch")) {
    const match = /v=([a-zA-Z0-9_-]+)/.exec(url);
    if (match?.[1]) {
      return `https://www.youtube.com/watch?v=${match[1]}`;
    }
  } else if (url.includes("youtu.be/")) {
    const match = /youtu\.be\/([a-zA-Z0-9_-]+)/.exec(url);
    if (match?.[1]) {
      return `https://www.youtube.com/watch?v=${match[1]}`;
    }
  }
  return url;
}

export function isYoutubeUrl(url: string): boolean {
  return YOUTUBE_PATTERNS.some((pattern) => pattern.test(url));
}

export function extractVideoId(url: string): string | null {
  const longMatch = /v=([a-zA-Z0-9_-]+)/.exec(url);
  if (longMatch?.[1]) {
    return longMatch[1];
  }

  const shortMatch = /youtu\.be\/([a-zA-Z0-9_-]+)/.exec(url);
  if (shortMatch?.[1]) {
    return shortMatch[1];
  }

  return null;
}
