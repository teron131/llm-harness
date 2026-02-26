function buildContextBlock(
  title?: string | null,
  description?: string | null,
): string {
  const metadataParts: string[] = [];
  if (title) {
    metadataParts.push(`Video Title: ${title}`);
  }
  if (description) {
    metadataParts.push(`Video Description: ${description}`);
  }

  if (metadataParts.length === 0) {
    return "";
  }

  return `\n# CONTEXTUAL INFORMATION:\n${metadataParts.join("\n")}\n`;
}

export function getGeminiSummaryPrompt(
  targetLanguage = "auto",
  title?: string | null,
  description?: string | null,
): string {
  const langDescriptions: Record<string, string> = {
    auto: "Use the same language as the video, or English if the language is unclear",
    en: "English (US)",
    "zh-TW": "Traditional Chinese (繁體中文)",
  };

  const langDesc = langDescriptions[targetLanguage] ?? targetLanguage;
  const instruction =
    targetLanguage === "auto"
      ? langDesc
      : `Write ALL output in ${langDesc}. Do not use English or any other language.`;

  const metadata = buildContextBlock(title, description);

  return [
    "Create a grounded, chronological summary.",
    metadata,
    `- OUTPUT LANGUAGE (REQUIRED): ${instruction}`,
    "",
    "SOURCE: You are given the full video. Use BOTH spoken content and visuals (on-screen text/slides/charts/code/UI). Do not invent details that are not clearly supported by what you can see/hear.",
    "",
    "Return JSON only (no extra text) with:",
    "- overview: string",
    "- chapters: array of { title: string, description: string, start_time?: string, end_time?: string }",
    "(start_time/end_time are optional MM:SS; omit if unsure)",
    "",
    "Rules:",
    "- Chapters must be chronological and non-overlapping",
    "- Avoid meta-language (no 'this video...' framing)",
    "- Exclude sponsors/promos/calls to action entirely",
  ].join("\n");
}

export function getLangchainSummaryPrompt(
  targetLanguage?: string | null,
  title?: string | null,
  description?: string | null,
): string {
  const metadata = buildContextBlock(title, description);

  const promptParts = [
    "Create a grounded, chronological summary of the transcript.",
    metadata,
    "Rules:",
    "- Ground every claim in the transcript; do not add unsupported details",
    "- Exclude sponsors/ads/promos/calls to action entirely",
    "- Avoid meta-language (no 'this video...', 'the speaker...', etc.)",
    "- Prefer concrete facts, names, numbers, and steps when present",
    "- Ensure output matches the provided response schema",
    "- Return JSON only with overview + chapters",
  ];

  if (targetLanguage) {
    promptParts.push(`\nOUTPUT LANGUAGE (REQUIRED): ${targetLanguage}`);
  }

  return promptParts.join("\n");
}

export function getGarbageFilterPrompt(): string {
  return [
    "Identify transcript lines that are NOT part of the core content and should be removed.",
    "Focus on: sponsors/ads/promos, discount codes, affiliate links, subscribe/like/call to action blocks, filler intros/outros, housekeeping, and other irrelevant segments.",
    "The transcript contains line tags like [L1], [L2], etc.",
    "Return ONLY the line ranges to remove (garbage_ranges).",
    "If unsure about a segment, prefer excluding it.",
  ].join("\n");
}

export function getQualityCheckPrompt(targetLanguage?: string | null): string {
  let systemPrompt = [
    "Evaluate the summary against the transcript.",
    "For each aspect in the response schema, return a rating (Fail/Refine/Pass) and a specific, actionable reason.",
    "Rules:",
    "- Be strict about transcript grounding",
    "- Treat any sponsor/promo/call to action content as a failure for no_garbage",
    "- Treat meta-language as a failure for meta_language_avoidance",
  ].join("\n");

  if (targetLanguage) {
    systemPrompt += `\nVerify the output language matches: ${targetLanguage}`;
  }

  return systemPrompt;
}
