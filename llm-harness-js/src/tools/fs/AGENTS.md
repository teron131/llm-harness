# Codemap: `llm-harness-js/src/tools/fs`

## Scope

- Filesystem sandbox operations and transcript tag-range transformations.

## What Module Is For

- This module provides sandboxed filesystem helpers and line-tag transcript filtering utilities.

## High-signal locations

- `src/tools/fs/fsTools.ts -> SandboxFS/makeFsTools`
- `src/tools/fs/fastCopy.ts -> TagRangeSchema/tagContent/filterContent/untagContent`

## Repository snapshot

- High-signal files listed below form the stable architecture anchors for this module.
- Keep imports and exports aligned with these anchors when extending behavior.

## Symbol Inventory

- Primary symbols are enumerated in the high-signal locations and syntax relationship sections.
- Preserve existing exported names unless changing a public contract intentionally.

## Key takeaways per location

- `SandboxFS.resolve` enforces path normalization and root containment.
- `makeFsTools` returns read/write/`ed`-edit functions for tool-calling workflows.
- `fastCopy` utilities are pure transforms used by YouTube summarizers.

## Project-specific conventions and rationale

- Keep traversal protection strict and centralized in `SandboxFS`.
- Preserve inclusive range removal semantics in `filterContent`.
- Keep text encoding behavior UTF-8 for read/write operations.

## Syntax Relationships

- `agents/youtube/summarizer*.ts -> import tagContent/filterContent/untagContent`
- `fsTools.ts -> spawn(\"ed\", ...)` for line-oriented edits

## General approach (not rigid checklist)

- Change path-validation rules only through `SandboxFS.resolve`.
- Keep tag/range utilities deterministic and side-effect free.
- Prefer small, explicit error messages for tool-call debugging.

## Validation commands

- `cd llm-harness-js && npm run build`
- `/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh`
