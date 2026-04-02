"""Standalone SQLite query and schema-navigation helpers."""

from __future__ import annotations

from datetime import date, datetime, time as datetime_time
from decimal import Decimal
from difflib import SequenceMatcher
from itertools import zip_longest
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, cast

from ..tabular.storage import SQLITE_CONTENTS_TABLE, SQLITE_SOURCES_TABLE, quote_identifier, sqlite_database_path

MAX_QUERY_ROWS = 200
MAX_DESCRIBE_SAMPLE_ROWS = 5
MAX_REPAIR_CANDIDATES = 3
MAX_TEXT_VALUE_HINTS = 5
MAX_SUGGESTED_TARGETS = 5
_MISSING = object()
_READ_ONLY_SQL_PREFIXES = ("SELECT", "WITH", "EXPLAIN")
_LEADING_SQL_COMMENT = re.compile(r"\A(?:\s+|--[^\n]*(?:\n|\Z)|/\*.*?\*/)*", re.DOTALL)
_TEXT_TYPE_MARKERS = ("CHAR", "CLOB", "TEXT", "VARCHAR")
_TEXT_HINT_NAME_MARKERS = ("account", "id", "name", "project", "region", "service", "sku")
_AMBIGUOUS_COLUMN_ERROR_PREFIX = "ambiguous column name:"
_MISSING_COLUMN_ERROR_PREFIX = "no such column:"
_MISSING_TABLE_ERROR_PREFIX = "no such table:"
_SUGGESTION_STOP_WORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "me",
    "of",
    "on",
    "show",
    "the",
    "to",
    "what",
    "which",
    "with",
}


def _jsonable_value(value: object) -> object:
    """Convert SQL query results into JSON-friendly values."""
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "hex": value.hex(),
            "length": len(value),
        }
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    return value


def _zip_exact(left: list[str], right: tuple[Any, ...]) -> list[tuple[str, Any]]:
    """Zip two sequences and fail if their lengths differ."""
    pairs: list[tuple[str, Any]] = []
    for left_item, right_item in zip_longest(left, right, fillvalue=_MISSING):
        if left_item is _MISSING or right_item is _MISSING:
            raise ValueError("SQL result row width did not match the reported column metadata.")
        pairs.append((cast(str, left_item), right_item))
    return pairs


def _error_result(
    *,
    database_path: str | Path | None,
    error_type: str,
    message: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build a stable error payload for tool callers."""
    payload: dict[str, Any] = {
        "status": "error",
        "error_type": error_type,
        "message": message,
    }
    if database_path is not None:
        payload["database_path"] = str(database_path)
    payload.update(extra)
    return payload


def _normalized_column_names(column_names: list[str | None]) -> tuple[list[str], list[str]]:
    """Return stable, unique column names for row dictionaries."""
    counts: dict[str, int] = {}
    seen: set[str] = set()
    normalized: list[str] = []
    originals: list[str] = []
    for index, raw_name in enumerate(column_names, start=1):
        base_name = raw_name or f"column_{index}"
        originals.append(base_name)
        occurrence = counts.get(base_name, 0)
        candidate_name = base_name
        while candidate_name in seen:
            occurrence += 1
            candidate_name = f"{base_name}__{occurrence + 1}"
        counts[base_name] = occurrence
        seen.add(candidate_name)
        normalized.append(candidate_name)
    return normalized, originals


def _leading_sql_keyword(sql: str) -> str:
    """Extract the first SQL keyword after whitespace and comments."""
    stripped = _LEADING_SQL_COMMENT.sub("", sql, count=1).lstrip()
    if not stripped:
        return ""
    return stripped.split(None, 1)[0].upper()


def _is_read_only_sql(sql: str) -> bool:
    """Return whether a SQL statement looks read-only."""
    return _leading_sql_keyword(sql) in _READ_ONLY_SQL_PREFIXES


def _sqlite_object_names(connection: sqlite3.Connection) -> set[str]:
    """Return all user-visible table and view names in a SQLite database."""
    return {
        cast(str, row[0])
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
    }


def _sample_rows(
    connection: sqlite3.Connection,
    *,
    target_name: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch a small row preview for one table or view."""
    safe_limit = max(0, min(limit, MAX_DESCRIBE_SAMPLE_ROWS))
    if safe_limit == 0:
        return []

    cursor = connection.execute(f"SELECT * FROM {quote_identifier(target_name)} LIMIT ?", [safe_limit])
    description = cursor.description or []
    column_names, _ = _normalized_column_names([cast(Any, column[0]) for column in description])
    return [{column_name: _jsonable_value(value) for column_name, value in _zip_exact(column_names, row)} for row in cursor.fetchall()]


def _fetch_catalog_metadata(connection: sqlite3.Connection) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Load shared tabular catalog metadata for list and suggest helpers."""
    content_rows = {
        cast(str, row[1]): {
            "content_id": cast(str, row[0]),
            "row_count": cast(int, row[2]),
        }
        for row in connection.execute(f"SELECT content_id, table_name, row_count FROM {SQLITE_CONTENTS_TABLE}").fetchall()
    }
    source_rows: dict[str, list[str]] = {}
    for row in connection.execute(f"SELECT content_id, source_path FROM {SQLITE_SOURCES_TABLE} ORDER BY source_path").fetchall():
        content_id = cast(str, row[0])
        source_rows.setdefault(content_id, []).append(cast(str, row[1]))
    return content_rows, source_rows


def _looks_like_text_column(column_name: str, declared_type: str) -> bool:
    """Return whether a column is a good candidate for distinct text-value hints."""
    normalized_type = declared_type.upper()
    normalized_name_tokens = set(re.findall(r"[a-z0-9]+", column_name.lower()))
    if any(marker in normalized_type for marker in _TEXT_TYPE_MARKERS):
        return True
    return any(marker in normalized_name_tokens for marker in _TEXT_HINT_NAME_MARKERS)


def _text_value_hints(
    connection: sqlite3.Connection,
    *,
    target_name: str,
    columns: list[dict[str, Any]],
    max_columns: int,
    max_values: int,
) -> dict[str, list[str]]:
    """Fetch a few distinct example values for useful text-like columns."""
    hints: dict[str, list[str]] = {}
    safe_max_columns = max(0, max_columns)
    safe_max_values = max(0, min(max_values, MAX_TEXT_VALUE_HINTS))
    if safe_max_columns == 0 or safe_max_values == 0:
        return hints

    candidate_columns = [
        cast(str, column["name"])
        for column in columns
        if _looks_like_text_column(cast(str, column["name"]), cast(str, column["type"]))
    ][:safe_max_columns]

    for column_name in candidate_columns:
        rows = connection.execute(
            f"""
            SELECT DISTINCT CAST({quote_identifier(column_name)} AS TEXT)
            FROM {quote_identifier(target_name)}
            WHERE {quote_identifier(column_name)} IS NOT NULL
              AND TRIM(CAST({quote_identifier(column_name)} AS TEXT)) != ''
            LIMIT ?
            """,
            [safe_max_values],
        ).fetchall()
        values = [cast(str, row[0]) for row in rows if row and row[0] is not None]
        if values:
            hints[column_name] = values
    return hints


def _tokenize_query(text: str) -> list[str]:
    """Tokenize a natural-language query for lightweight target suggestion."""
    tokens = [token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) >= 2]
    return [token for token in tokens if token not in _SUGGESTION_STOP_WORDS]


def _identifier_tokens(value: str) -> set[str]:
    """Split one identifier into comparable lowercase tokens."""
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def _identifier_similarity(reference: str, candidate: str) -> float:
    """Score one candidate identifier against a missing identifier."""
    normalized_reference = re.sub(r"[^a-z0-9]+", "", reference.lower())
    normalized_candidate = re.sub(r"[^a-z0-9]+", "", candidate.lower())
    if not normalized_reference or not normalized_candidate:
        return 0.0
    if normalized_reference == normalized_candidate:
        return 100.0

    score = 0.0
    if normalized_reference in normalized_candidate or normalized_candidate in normalized_reference:
        score += 40.0

    reference_tokens = _identifier_tokens(reference)
    candidate_tokens = _identifier_tokens(candidate)
    if reference_tokens and candidate_tokens:
        shared_tokens = reference_tokens & candidate_tokens
        score += 20.0 * len(shared_tokens)
        if shared_tokens == reference_tokens == candidate_tokens:
            score += 20.0

    score += SequenceMatcher(a=normalized_reference, b=normalized_candidate).ratio() * 20.0
    return score


def _rank_identifier_candidates(identifier: str, candidates: list[str], *, max_matches: int) -> list[str]:
    """Return the best schema identifier matches for a missing target or column."""
    if not identifier or max_matches <= 0:
        return []

    scored_candidates = sorted(
        (
            (_identifier_similarity(identifier, candidate), candidate)
            for candidate in dict.fromkeys(candidates)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return [candidate for score, candidate in scored_candidates if score > 0][:max_matches]


def _error_identifier(error_message: str, prefix: str) -> str:
    """Extract the identifier payload from a SQLite error message."""
    identifier = error_message[len(prefix) :].strip()
    if not identifier:
        return ""
    unqualified = identifier.rsplit(".", 1)[-1]
    return unqualified.strip('"`[]')


def _format_repair_candidates(
    candidates: list[dict[str, Any]],
    *,
    include_targets: bool,
) -> str:
    """Format repair candidates into one compact human-readable string."""
    parts = []
    for candidate in candidates:
        name = cast(str, candidate["name"])
        if include_targets:
            targets = cast(list[str], candidate.get("targets", []))
            target_suffix = f" on {', '.join(targets)}" if targets else ""
            parts.append(f"{name}{target_suffix}")
        else:
            parts.append(name)
    return ", ".join(parts)


def _target_search_text(
    *,
    name: str,
    target_type: str,
    kind: str,
    column_names: list[str],
    source_paths: list[str],
    create_sql: str | None,
) -> str:
    """Build a search blob for one target."""
    parts = [
        name.replace("_", " "),
        target_type,
        kind.replace("_", " "),
        " ".join(column_names),
        " ".join(source_paths),
        create_sql or "",
    ]
    return " ".join(parts).lower()


def _target_score(
    *,
    tokens: list[str],
    name: str,
    column_names: list[str],
    source_paths: list[str],
    search_text: str,
) -> tuple[int, list[str]]:
    """Score one target against a lightweight NL query."""
    score = 0
    reasons: list[str] = []
    lowered_name = name.lower()
    lowered_columns = [column.lower() for column in column_names]
    lowered_sources = [source.lower() for source in source_paths]

    for token in tokens:
        if token in lowered_name:
            score += 5
            reasons.append(f"name matched '{token}'")
            continue
        matching_columns = [column for column in lowered_columns if token in column]
        if matching_columns:
            score += 3
            reasons.append(f"column matched '{token}'")
            continue
        if any(token in source for source in lowered_sources):
            score += 2
            reasons.append(f"source matched '{token}'")
            continue
        if token in search_text:
            score += 1
            reasons.append(f"context matched '{token}'")

    if tokens and all(token in lowered_name for token in tokens):
        score += 2
        reasons.append("all tokens matched target name")
    return score, reasons


def _kind_bias(kind: str) -> int:
    """Prefer stable views over raw storage artifacts during target suggestion."""
    if kind == "typed_content_view":
        return 3
    if kind == "view_or_table":
        return 1
    if kind == "raw_content_table":
        return -2
    if kind == "internal_catalog":
        return -5
    return 0


def resolve_db_path(
    *,
    root_dir: str | Path | None = None,
    database_path: str | Path | None = None,
) -> Path:
    """Resolve the SQLite database path and ensure it exists."""
    resolved_path = sqlite_database_path(root_dir=root_dir) if database_path is None else Path(database_path)
    if not resolved_path.exists():
        raise ValueError(f"SQLite database does not exist: {resolved_path}")
    return resolved_path


def classify_target(name: str) -> str:
    """Classify a SQLite target using current naming conventions."""
    if name in (SQLITE_CONTENTS_TABLE, SQLITE_SOURCES_TABLE):
        return "internal_catalog"
    if name.startswith("content_") and name.endswith("_typed"):
        return "typed_content_view"
    if name.startswith("content_"):
        return "raw_content_table"
    return "view_or_table"


def run_sql(
    sql: str,
    *,
    root_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    max_rows: int = MAX_QUERY_ROWS,
) -> dict[str, Any]:
    """Run SQL against a SQLite database and return a bounded result."""
    requested_path = sqlite_database_path(root_dir=root_dir) if database_path is None else Path(database_path)
    safe_max_rows = max(1, max_rows)
    connection: sqlite3.Connection | None = None
    try:
        if not sql.strip():
            return _error_result(
                database_path=requested_path,
                error_type="empty_sql",
                message="SQL query must not be empty.",
                max_rows=safe_max_rows,
            )
        # This is a quick UX guard. The read-only SQLite connection is the actual safety boundary.
        if not _is_read_only_sql(sql):
            return _error_result(
                database_path=requested_path,
                error_type="disallowed_sql",
                message="Only read-only SELECT, WITH, and EXPLAIN queries are allowed.",
                max_rows=safe_max_rows,
            )

        resolved_path = resolve_db_path(
            root_dir=root_dir,
            database_path=database_path,
        )
        connection = sqlite3.connect(f"file:{resolved_path}?mode=ro", uri=True)
        cursor = connection.execute(sql)
        description = cursor.description
        if not description:
            return {
                "database_path": str(resolved_path),
                "status": "ok",
                "max_rows": safe_max_rows,
                "row_count": 0,
                "truncated": False,
                "columns": [],
                "rows": [],
            }

        column_names, original_columns = _normalized_column_names([cast(Any, column[0]) for column in description])
        raw_rows = cursor.fetchmany(safe_max_rows + 1)
        truncated = len(raw_rows) > safe_max_rows
        result_rows = raw_rows[:safe_max_rows]
        rows = [{column_name: _jsonable_value(value) for column_name, value in _zip_exact(column_names, row)} for row in result_rows]

        payload = {
            "database_path": str(resolved_path),
            "status": "ok",
            "max_rows": safe_max_rows,
            "row_count": len(rows),
            "truncated": truncated,
            "columns": column_names,
            "rows": rows,
        }
        if column_names != original_columns:
            payload["original_columns"] = original_columns
        return payload
    except ValueError as exc:
        return _error_result(
            database_path=requested_path,
            error_type="missing_database",
            message=str(exc),
            max_rows=safe_max_rows,
        )
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return _error_result(
            database_path=requested_path,
            error_type="sql_execution_error",
            message=str(exc),
            max_rows=safe_max_rows,
        )
    finally:
        if connection is not None:
            connection.close()


def list_targets(
    *,
    root_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    include_internal: bool = False,
) -> dict[str, Any]:
    """List queryable SQLite tables and views."""
    requested_path = sqlite_database_path(root_dir=root_dir) if database_path is None else Path(database_path)
    connection: sqlite3.Connection | None = None
    try:
        resolved_path = resolve_db_path(
            root_dir=root_dir,
            database_path=database_path,
        )
        connection = sqlite3.connect(f"file:{resolved_path}?mode=ro", uri=True)
        object_names = _sqlite_object_names(connection)
        targets = connection.execute(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        has_catalog = SQLITE_CONTENTS_TABLE in object_names and SQLITE_SOURCES_TABLE in object_names
        content_rows: dict[str, dict[str, Any]] = {}
        source_rows: dict[str, list[str]] = {}
        if has_catalog:
            content_rows, source_rows = _fetch_catalog_metadata(connection)

        items = []
        for row in targets:
            name = cast(str, row[0])
            target_type = cast(str, row[1])
            kind = classify_target(name)
            if not include_internal and kind == "internal_catalog":
                continue

            base_table_name = name.removesuffix("_typed") if kind == "typed_content_view" else name
            content_metadata = content_rows.get(base_table_name)
            source_paths = source_rows.get(content_metadata["content_id"], []) if content_metadata else []

            items.append(
                {
                    "name": name,
                    "type": target_type,
                    "kind": kind,
                    "row_count": content_metadata["row_count"] if content_metadata else None,
                    "source_paths": source_paths,
                }
            )

        return {
            "database_path": str(resolved_path),
            "status": "ok",
            "has_tabular_catalog": has_catalog,
            "target_count": len(items),
            "targets": items,
        }
    except ValueError as exc:
        return _error_result(
            database_path=requested_path,
            error_type="missing_database",
            message=str(exc),
        )
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return _error_result(
            database_path=requested_path,
            error_type="sql_execution_error",
            message=str(exc),
        )
    finally:
        if connection is not None:
            connection.close()


def describe_target(
    target_name: str,
    *,
    root_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    sample_rows: int = 3,
    text_value_hints: int = 3,
) -> dict[str, Any]:
    """Describe a single SQLite table or view."""
    requested_path = sqlite_database_path(root_dir=root_dir) if database_path is None else Path(database_path)
    connection: sqlite3.Connection | None = None
    try:
        resolved_path = resolve_db_path(
            root_dir=root_dir,
            database_path=database_path,
        )
        connection = sqlite3.connect(f"file:{resolved_path}?mode=ro", uri=True)
        object_names = _sqlite_object_names(connection)
        master_row = connection.execute(
            """
            SELECT name, type, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view') AND name = ?
            """,
            [target_name],
        ).fetchone()
        if master_row is None:
            return _error_result(
                database_path=resolved_path,
                error_type="missing_target",
                message=f"SQLite target does not exist: {target_name}",
                target_name=target_name,
            )

        name = cast(str, master_row[0])
        target_type = cast(str, master_row[1])
        create_sql = cast(Any, master_row[2])
        kind = classify_target(name)
        pragma_rows = connection.execute(f"PRAGMA table_info({quote_identifier(name)})").fetchall()
        columns = [
            {
                "name": cast(str, row[1]),
                "type": cast(str, row[2]),
                "not_null": bool(row[3]),
                "default_value": row[4],
                "primary_key_position": cast(int, row[5]),
            }
            for row in pragma_rows
        ]
        sample_row_items = _sample_rows(
            connection,
            target_name=name,
            limit=sample_rows,
        )
        text_value_hint_map = _text_value_hints(
            connection,
            target_name=name,
            columns=columns,
            max_columns=text_value_hints,
            max_values=MAX_TEXT_VALUE_HINTS,
        )

        has_catalog = SQLITE_CONTENTS_TABLE in object_names and SQLITE_SOURCES_TABLE in object_names
        source_mappings = []
        content_id = None
        content_schema = None
        row_count = None
        if has_catalog:
            base_table_name = name.removesuffix("_typed") if kind == "typed_content_view" else name
            catalog_row = connection.execute(
                f"""
                SELECT content_id, source_format, row_count, column_schema_json
                FROM {SQLITE_CONTENTS_TABLE}
                WHERE table_name = ?
                """,
                [base_table_name],
            ).fetchone()

            if catalog_row is not None:
                content_id = cast(str, catalog_row[0])
                row_count = cast(int, catalog_row[2])
                content_schema = json.loads(cast(str, catalog_row[3]))
                source_mappings = [
                    {
                        "source_path": cast(str, row[0]),
                        "source_format": cast(str, row[1]),
                        "source_sheet_name": cast(str, row[2]),
                        "source_table_name": cast(str, row[3]),
                        "fast_fingerprint": cast(str, row[4]),
                    }
                    for row in connection.execute(
                        f"""
                        SELECT source_path, source_format, source_sheet_name, source_table_name, fast_fingerprint
                        FROM {SQLITE_SOURCES_TABLE}
                        WHERE content_id = ?
                        ORDER BY source_path, source_table_name
                        """,
                        [content_id],
                    ).fetchall()
                ]

        return {
            "database_path": str(resolved_path),
            "status": "ok",
            "has_tabular_catalog": has_catalog,
            "name": name,
            "type": target_type,
            "kind": kind,
            "row_count": row_count,
            "columns": columns,
            "sample_rows": sample_row_items,
            "text_value_hints": text_value_hint_map,
            "create_sql": create_sql,
            "content_id": content_id,
            "content_schema": content_schema,
            "source_mappings": source_mappings,
        }
    except ValueError as exc:
        return _error_result(
            database_path=requested_path,
            error_type="missing_database",
            message=str(exc),
            target_name=target_name,
        )
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return _error_result(
            database_path=requested_path,
            error_type="sql_execution_error",
            message=str(exc),
            target_name=target_name,
        )
    finally:
        if connection is not None:
            connection.close()


def suggest_targets(
    question: str,
    *,
    root_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    include_internal: bool = False,
    max_results: int = MAX_SUGGESTED_TARGETS,
) -> dict[str, Any]:
    """Suggest likely tables or views for a natural-language question."""
    requested_path = sqlite_database_path(root_dir=root_dir) if database_path is None else Path(database_path)
    connection: sqlite3.Connection | None = None
    safe_max_results = max(1, max_results)
    try:
        tokens = _tokenize_query(question)
        if not tokens:
            return _error_result(
                database_path=requested_path,
                error_type="empty_question",
                message="Question must include at least one meaningful search token.",
                max_results=safe_max_results,
            )

        resolved_path = resolve_db_path(
            root_dir=root_dir,
            database_path=database_path,
        )
        connection = sqlite3.connect(f"file:{resolved_path}?mode=ro", uri=True)
        object_names = _sqlite_object_names(connection)
        has_catalog = SQLITE_CONTENTS_TABLE in object_names and SQLITE_SOURCES_TABLE in object_names

        content_rows: dict[str, dict[str, Any]] = {}
        source_rows: dict[str, list[str]] = {}
        if has_catalog:
            content_rows, source_rows = _fetch_catalog_metadata(connection)

        suggestions = []
        master_rows = connection.execute(
            """
            SELECT name, type, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        for master_row in master_rows:
            name = cast(str, master_row[0])
            target_type = cast(str, master_row[1])
            create_sql = cast(Any, master_row[2])
            kind = classify_target(name)
            if not include_internal and kind == "internal_catalog":
                continue

            base_table_name = name.removesuffix("_typed") if kind == "typed_content_view" else name
            content_metadata = content_rows.get(base_table_name)
            source_paths = source_rows.get(content_metadata["content_id"], []) if content_metadata else []
            pragma_rows = connection.execute(f"PRAGMA table_info({quote_identifier(name)})").fetchall()
            column_names = [cast(str, row[1]) for row in pragma_rows]
            search_text = _target_search_text(
                name=name,
                target_type=target_type,
                kind=kind,
                column_names=column_names,
                source_paths=source_paths,
                create_sql=create_sql,
            )
            score, reasons = _target_score(
                tokens=tokens,
                name=name,
                column_names=column_names,
                source_paths=source_paths,
                search_text=search_text,
            )
            if score <= 0:
                continue
            score += _kind_bias(kind)

            suggestions.append(
                {
                    "name": name,
                    "type": target_type,
                    "kind": kind,
                    "score": score,
                    "reasons": reasons,
                    "columns": column_names,
                    "source_paths": source_paths,
                    "row_count": content_metadata["row_count"] if content_metadata else None,
                }
            )

        suggestions.sort(key=lambda item: (-cast(int, item["score"]), cast(str, item["name"])))
        return {
            "database_path": str(resolved_path),
            "status": "ok",
            "has_tabular_catalog": has_catalog,
            "question": question,
            "tokens": tokens,
            "suggestion_count": len(suggestions[:safe_max_results]),
            "suggestions": suggestions[:safe_max_results],
        }
    except ValueError as exc:
        return _error_result(
            database_path=requested_path,
            error_type="missing_database",
            message=str(exc),
            question=question,
            max_results=safe_max_results,
        )
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return _error_result(
            database_path=requested_path,
            error_type="sql_execution_error",
            message=str(exc),
            question=question,
            max_results=safe_max_results,
        )
    finally:
        if connection is not None:
            connection.close()


def suggest_sql_error_repair(
    error_message: str,
    *,
    available_targets: list[str],
    target_columns: dict[str, list[str]],
    max_matches: int = MAX_REPAIR_CANDIDATES,
) -> list[dict[str, Any]]:
    """Return deterministic schema-aware repair hints for common SQLite errors."""
    safe_max_matches = max(1, max_matches)
    lowered_error = error_message.strip().lower()

    if lowered_error.startswith(_MISSING_COLUMN_ERROR_PREFIX):
        missing_column = _error_identifier(error_message, _MISSING_COLUMN_ERROR_PREFIX)
        columns_by_name: dict[str, list[str]] = {}
        for target_name, columns in target_columns.items():
            for column_name in columns:
                columns_by_name.setdefault(column_name, []).append(target_name)

        candidate_columns = _rank_identifier_candidates(
            missing_column,
            list(columns_by_name),
            max_matches=safe_max_matches,
        )
        if not candidate_columns:
            return []

        candidates = [
            {
                "name": column_name,
                "targets": sorted(columns_by_name[column_name]),
            }
            for column_name in candidate_columns
        ]
        return [
            {
                "kind": "missing_column",
                "identifier": missing_column,
                "candidates": candidates,
                "message": (
                    f"Column `{missing_column}` was not found. "
                    f"Closest inspected columns: {_format_repair_candidates(candidates, include_targets=True)}."
                ),
            }
        ]

    if lowered_error.startswith(_MISSING_TABLE_ERROR_PREFIX):
        missing_target = _error_identifier(error_message, _MISSING_TABLE_ERROR_PREFIX)
        candidate_targets = _rank_identifier_candidates(
            missing_target,
            available_targets,
            max_matches=safe_max_matches,
        )
        if not candidate_targets:
            return []

        candidates = [{"name": target_name} for target_name in candidate_targets]
        return [
            {
                "kind": "missing_target",
                "identifier": missing_target,
                "candidates": candidates,
                "message": (
                    f"Target `{missing_target}` was not found. "
                    f"Closest inspected targets: {_format_repair_candidates(candidates, include_targets=False)}."
                ),
            }
        ]

    if lowered_error.startswith(_AMBIGUOUS_COLUMN_ERROR_PREFIX):
        ambiguous_column = _error_identifier(error_message, _AMBIGUOUS_COLUMN_ERROR_PREFIX)
        matching_targets = sorted(
            target_name for target_name, columns in target_columns.items() if ambiguous_column in columns
        )
        if not matching_targets:
            return []

        candidates = [{"name": ambiguous_column, "targets": matching_targets}]
        return [
            {
                "kind": "ambiguous_column",
                "identifier": ambiguous_column,
                "candidates": candidates,
                "message": (
                    f"Column `{ambiguous_column}` is ambiguous. "
                    f"Qualify it with one of: {', '.join(matching_targets)}."
                ),
            }
        ]

    return []
