import { z } from "zod";

import { TagRangeSchema } from "../../tools/fs/fast-copy.js";
import { s2hk } from "../../utils/text-utils.js";

export const ChapterSchema = z.object({
  title: z.string().transform((value) => s2hk(value)),
  description: z.string().transform((value) => s2hk(value)),
  start_time: z.string().optional().nullable(),
  end_time: z.string().optional().nullable(),
});

export const SummarySchema = z.object({
  overview: z.string().transform((value) => s2hk(value)),
  chapters: z.array(ChapterSchema).min(1),
});

export type Chapter = z.output<typeof ChapterSchema>;
export type Summary = z.output<typeof SummarySchema>;

export function summaryToText(summary: Summary): string {
  const lines = [
    "=".repeat(80),
    "SUMMARY:",
    "=".repeat(80),
    `\nOverview:\n${summary.overview}`,
    `\nChapters (${summary.chapters.length}):`,
  ];

  summary.chapters.forEach((chapter, i) => {
    lines.push(`\n  Chapter ${i + 1}: ${chapter.title}`);
    lines.push(`    Summary: ${chapter.description}`);
    if (chapter.start_time || chapter.end_time) {
      lines.push(
        `    Time: ${chapter.start_time ?? "?"} - ${chapter.end_time ?? "?"}`,
      );
    }
  });

  return lines.join("\n");
}

export const GarbageIdentificationSchema = z.object({
  garbage_ranges: z.array(TagRangeSchema),
});

export const RateSchema = z.object({
  rate: z.enum(["Fail", "Refine", "Pass"]),
  reason: z.string(),
});

export const QualitySchema = z.object({
  completeness: RateSchema,
  structure: RateSchema,
  no_garbage: RateSchema,
  meta_language_avoidance: RateSchema,
  useful_keywords: RateSchema,
  correct_language: RateSchema,
});

export type Quality = z.infer<typeof QualitySchema>;

export function allAspects(quality: Quality) {
  return [
    quality.completeness,
    quality.structure,
    quality.no_garbage,
    quality.meta_language_avoidance,
    quality.useful_keywords,
    quality.correct_language,
  ];
}

export function percentageScore(quality: Quality): number {
  const aspects = allAspects(quality);
  const passCount = aspects.filter((x) => x.rate === "Pass").length;
  const refineCount = aspects.filter((x) => x.rate === "Refine").length;
  return Math.floor((passCount * 100 + refineCount * 50) / aspects.length);
}

export function isAcceptable(quality: Quality): boolean {
  return percentageScore(quality) >= 80;
}
