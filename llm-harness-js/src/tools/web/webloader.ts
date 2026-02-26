import TurndownService from "turndown";

function cleanMarkdown(markdown: string): string {
  return markdown
    .replace(/^<!-- image -->$/gm, "")
    .replace(/ {2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n");
}

async function convertUrl(url: string): Promise<string | null> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return null;
    }

    const html = await response.text();
    const turndown = new TurndownService();
    const markdown = turndown.turndown(html);
    return markdown ? cleanMarkdown(markdown) : null;
  } catch {
    return null;
  }
}

export async function webloader(
  urls: string | string[],
): Promise<Array<string | null>> {
  const normalizedUrls = Array.isArray(urls) ? urls : [urls];
  return Promise.all(normalizedUrls.map((url) => convertUrl(url)));
}

export async function webloaderTool(
  urls: string[],
): Promise<Array<string | null>> {
  return webloader(urls);
}
