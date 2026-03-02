import { tool } from "@langchain/core/tools";
import TurndownService from "turndown";
import { z } from "zod";

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

export function webloader(
  urls: string | string[],
): Promise<Array<string | null>> {
  const normalizedUrls = Array.isArray(urls) ? urls : [urls];
  return Promise.all(normalizedUrls.map((url) => convertUrl(url)));
}

export const webloaderTool = tool(
  async ({ urls }: { urls: string[] }): Promise<Array<string | null>> =>
    webloader(urls),
  {
    name: "webloader_tool",
    description: "Load the web content from the given URLs.",
    schema: z.object({
      urls: z.array(z.string()),
    }),
  },
);
