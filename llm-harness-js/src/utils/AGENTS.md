# Codemap: `llm-harness-js/src/utils`

## Scope

- Pure helper utilities for text conversion, image encoding, and YouTube URL/text normalization.

## High-signal locations

- `src/utils/youtubeUtils.ts -> cleanText/cleanYoutubeUrl/isYoutubeUrl/extractVideoId`
- `src/utils/textUtils.ts -> s2hk`
- `src/utils/imageUtils.ts -> loadImageBase64`
- `src/utils/index.ts -> utility export surface`

## Key takeaways per location

- `youtubeUtils.ts` centralizes URL detection/normalization and transcript text cleanup.
- `textUtils.ts` caches OpenCC converter initialization and exposes `s2hk`.
- `imageUtils.ts` loads local/remote images, resizes proportionally, and emits base64.
- `index.ts` is the stable import surface consumed by higher modules.

## Project-specific conventions and rationale

- Keep this layer deterministic and side-effect light.
- Reuse centralized URL/text helpers in tools/agents to avoid regex duplication.
- Preserve default image behavior (`maxSize`, output format) unless explicitly changed.

## Syntax relationship highlights (ast-grep-first)

- `tools/youtube/scraper.ts -> imports all youtube utils`
- `agents/youtube/schemas.ts -> imports s2hk`
- `tools` and `agents` import utilities through `utils/index.ts` or specific modules.

## General approach (not rigid checklist)

- Add new helpers here only when shared by multiple higher-level modules.
- Prefer small, explicit transforms over hidden implicit behavior.

## Validation commands

- `cd llm-harness-js && npm run build`
- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
