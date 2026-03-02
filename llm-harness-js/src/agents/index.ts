import { webloaderTool } from "../tools/web/index.js";
import { youtubeLoader } from "./youtube/index.js";

export function youtubeloaderTool(url: string): Promise<string> {
  return youtubeLoader(url);
}

export function getTools() {
  return [webloaderTool, youtubeloaderTool];
}

export { webloader, webloaderTool } from "../tools/web/index.js";
export {
  BaseHarnessAgent,
  ExaAgent,
  ImageAnalysisAgent,
  WebLoaderAgent,
  WebSearchAgent,
  WebSearchLoaderAgent,
  YouTubeSummarizer,
  YouTubeSummarizerGemini,
  YouTubeSummarizerGeminiAgent,
  YouTubeSummarizerLiteAgent,
  YouTubeSummarizerReAct,
  YouTubeSummarizerReActAgent,
} from "./agents.js";
export { youtubeLoader } from "./youtube/index.js";
