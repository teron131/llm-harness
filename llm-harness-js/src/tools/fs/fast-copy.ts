import { z } from "zod";

export const TagRangeSchema = z.object({
  start_tag: z.string().describe("The starting line tag, e.g., [L10]"),
  end_tag: z.string().describe("The ending line tag, e.g., [L20]"),
});

export type TagRange = z.infer<typeof TagRangeSchema>;

export function tagContent(text: string): string {
  return text
    .split(/\r?\n/)
    .map((line, index) => `[L${index + 1}] ${line}`)
    .join("\n");
}

export function untagContent(text: string): string {
  return text.replace(/^\[L\d+\]\s*/gm, "");
}

export function filterContent(taggedText: string, ranges: TagRange[]): string {
  const lines = taggedText.split(/\r?\n/);
  if (lines.length === 0 || ranges.length === 0) {
    return taggedText;
  }

  const tagToIdx = new Map<string, number>();
  for (const [idx, line] of lines.entries()) {
    if (!line.startsWith("[L")) {
      continue;
    }

    const end = line.indexOf("]");
    if (end !== -1) {
      tagToIdx.set(line.slice(0, end + 1), idx);
    }
  }

  const keepMask = Array.from({ length: lines.length }, () => true);

  for (const range of ranges) {
    const startIdx = tagToIdx.get(range.start_tag);
    const endIdx = tagToIdx.get(range.end_tag);
    if (startIdx === undefined || endIdx === undefined) {
      continue;
    }

    const [firstIdx, lastIdx] =
      startIdx <= endIdx ? [startIdx, endIdx] : [endIdx, startIdx];
    for (let i = firstIdx; i <= lastIdx; i += 1) {
      keepMask[i] = false;
    }
  }

  return lines.filter((_, idx) => keepMask[idx]).join("\n");
}
