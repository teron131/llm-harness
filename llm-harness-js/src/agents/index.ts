import { webloader, webloaderTool } from "../tools/web/index.js";
import { youtubeLoader } from "./youtube/index.js";

import {
  BaseHarnessAgent,
  ExaAgent,
  ImageAnalysisAgent,
  WebLoaderAgent,
  WebSearchAgent,
  WebSearchLoaderAgent,
  YouTubeSummarizerGeminiAgent,
  YouTubeSummarizerLiteAgent,
  YouTubeSummarizerReActAgent,
} from "./agents.js";

export function youtubeloaderTool(url: string): Promise<string> {
  return youtubeLoader(url);
}

export function getTools() {
  return [webloaderTool, youtubeloaderTool];
}

export {
  BaseHarnessAgent,
  ExaAgent,
  ImageAnalysisAgent,
  WebLoaderAgent,
  WebSearchAgent,
  WebSearchLoaderAgent,
  YouTubeSummarizerGeminiAgent,
  YouTubeSummarizerLiteAgent,
  YouTubeSummarizerReActAgent,
  webloader,
  webloaderTool,
  youtubeLoader,
};
