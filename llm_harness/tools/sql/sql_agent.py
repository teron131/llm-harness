"""Minimal LangGraph orchestration for SQL question answering."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import re
from typing import Any, Literal, cast

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from ...clients.openrouter import ChatOpenRouter
from .query import describe_target, run_sql, suggest_sql_error_repair, suggest_targets

DEFAULT_SQL_AGENT_MODEL_ENV = "FAST_LLM"
DEFAULT_SQL_AGENT_MODEL = "openai/gpt-5.4-nano"
DEFAULT_SQL_AGENT_REASONING: Literal["minimal", "low", "medium", "high"] = "medium"
MAX_INSPECTED_TARGETS = 2
_QUESTION_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "by",
    "for",
    "get",
    "give",
    "how",
    "i",
    "in",
    "is",
    "me",
    "of",
    "our",
    "show",
    "tell",
    "the",
    "to",
    "us",
    "we",
    "what",
    "which",
    "with",
}
_VAGUE_QUESTION_TOKENS = {"doing", "everything", "numbers", "overall", "overview", "performance", "stats", "status", "summary"}


class SQLPlan(BaseModel):
    """Structured SQL planning output."""

    ready: bool = Field(description="Whether the planner is confident enough to run SQL.")
    sql: str | None = Field(default=None, description="Read-only SQL query to execute when ready is true.")
    selected_targets: list[str] = Field(default_factory=list, description="Target tables or views used by the plan.")
    rationale: str = Field(default="", description="Short reasoning for the chosen SQL.")
    blocking_reason: str | None = Field(default=None, description="Why planning could not safely proceed.")


class SQLAgentInput(BaseModel):
    """Public input schema for the SQL agent graph."""

    question: str
    database_path: str | None = None
    max_suggestions: int = 3
    max_repairs: int = 2
    sample_rows: int = 3
    text_value_hints: int = 3


class SQLAgentOutput(BaseModel):
    """Public output schema for the SQL agent graph."""

    status: str = "pending"
    selected_targets: list[str] = Field(default_factory=list)
    candidate_sql: str | None = None
    repair_hints: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    attempts: int = 0
    rationale: str | None = None
    last_error: str | None = None
    trace: list[str] = Field(default_factory=list)


class SQLAgentState(SQLAgentInput, SQLAgentOutput):
    """Internal graph state."""

    suggestions: list[dict[str, Any]] = Field(default_factory=list)
    inspected_targets: list[dict[str, Any]] = Field(default_factory=list)
    plan: SQLPlan | None = None
    repair_count: int = 0


PlannerFn = Callable[[SQLAgentState], SQLPlan]

SQL_PLANNER_SYSTEM_PROMPT = """You are a careful SQLite analyst.

Turn the user question into one read-only SQLite query.

Rules:
- Use only SELECT, WITH, or EXPLAIN.
- Use only tables/views and columns that appear in the inspected target context.
- Prefer curated views and stable business-facing targets when possible.
- If the question is concrete but omits a time range, use all available data by default.
- If the question uses a loose business synonym, choose the most natural business-facing entity from the inspected context instead of blocking.
- If the question is vague or under-specified, set ready=false and ask for a more concrete metric or grouping.
- If the inspected context is not enough, set ready=false instead of guessing.
- If there was a previous SQL error, fix the query directly and avoid repeating the same mistake.
- If repair_hints are present, prefer those exact replacement identifiers or target names.
- Keep the SQL general-purpose and minimal.
"""


def _append_trace(state: SQLAgentState, message: str) -> list[str]:
    """Append one trace message."""
    return [*state.trace, message]


def _needs_clarification(question: str) -> str | None:
    """Return a clarification message when the question is too vague."""
    tokens = [token for token in re.findall(r"[a-z0-9_]+", question.lower()) if token not in _QUESTION_STOP_WORDS]
    if not tokens:
        return "Question is too vague. Ask for a specific metric, dimension, or summary target."
    if len(tokens) <= 2 and all(token in _VAGUE_QUESTION_TOKENS for token in tokens):
        return "Question is too vague. Ask for a concrete metric or breakdown, for example revenue by customer or gross profit summary."
    return None


def _build_planner_messages(state: SQLAgentState) -> list[SystemMessage | HumanMessage]:
    """Build planner messages for the structured planning model."""
    payload = {
        "question": state.question,
        "candidate_targets": state.suggestions,
        "inspected_targets": [
            {
                "name": target.get("name"),
                "kind": target.get("kind"),
                "type": target.get("type"),
                "row_count": target.get("row_count"),
                "columns": [
                    {
                        "name": column.get("name"),
                        "type": column.get("type"),
                    }
                    for column in target.get("columns", [])
                ],
                "sample_rows": target.get("sample_rows", []),
                "text_value_hints": target.get("text_value_hints", {}),
                "source_mappings": target.get("source_mappings", []),
            }
            for target in state.inspected_targets
        ],
        "previous_sql": state.candidate_sql,
        "previous_error": state.last_error,
        "repair_hints": state.repair_hints,
        "repair_count": state.repair_count,
    }
    return [
        SystemMessage(content=SQL_PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=True, sort_keys=True)),
    ]


def make_llm_planner(
    llm: BaseChatModel | None = None,
    *,
    model: str | None = None,
    temperature: float = 0,
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = DEFAULT_SQL_AGENT_REASONING,
) -> PlannerFn:
    """Create a structured SQL planner backed by a chat model."""
    if llm is None:
        resolved_model = model or os.getenv(DEFAULT_SQL_AGENT_MODEL_ENV) or DEFAULT_SQL_AGENT_MODEL
        if not resolved_model:
            raise ValueError(f"No SQL agent model configured. Pass `llm=...`, `model=...`, or set `{DEFAULT_SQL_AGENT_MODEL_ENV}`.")
        llm = ChatOpenRouter(
            model=resolved_model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

    structured_llm = llm.with_structured_output(SQLPlan)

    def planner(state: SQLAgentState) -> SQLPlan:
        return structured_llm.invoke(_build_planner_messages(state))

    return planner


def suggest_node(state: SQLAgentState) -> dict[str, Any]:
    """Suggest likely SQL targets for the question."""
    suggestion_result = suggest_targets(
        state.question,
        database_path=state.database_path,
        max_results=state.max_suggestions,
    )
    if suggestion_result["status"] != "ok":
        return {
            "status": "error",
            "last_error": suggestion_result["message"],
            "result": suggestion_result,
            "trace": _append_trace(state, f"suggest failed: {suggestion_result['message']}"),
        }

    return {
        "status": "suggested",
        "suggestions": suggestion_result["suggestions"],
        "trace": _append_trace(state, f"suggested {suggestion_result['suggestion_count']} targets"),
    }


def inspect_node(state: SQLAgentState) -> dict[str, Any]:
    """Inspect the top suggested targets before planning SQL."""
    inspected_targets: list[dict[str, Any]] = []
    for suggestion in state.suggestions[:MAX_INSPECTED_TARGETS]:
        description = describe_target(
            cast(str, suggestion["name"]),
            database_path=state.database_path,
            sample_rows=state.sample_rows,
            text_value_hints=state.text_value_hints,
        )
        if description["status"] == "ok":
            inspected_targets.append(description)

    if not inspected_targets:
        return {
            "status": "error",
            "last_error": "No inspectable SQL targets were available.",
            "trace": _append_trace(state, "inspect found no usable targets"),
        }

    return {
        "status": "inspected",
        "inspected_targets": inspected_targets,
        "trace": _append_trace(state, f"inspected {len(inspected_targets)} targets"),
    }


def clarify_node(state: SQLAgentState) -> dict[str, Any]:
    """Block execution when the question needs clarification."""
    message = _needs_clarification(state.question) or "Question needs clarification before SQL planning."
    return {
        "status": "blocked",
        "last_error": message,
        "trace": _append_trace(state, "blocked for clarification"),
    }


def make_plan_node(planner: PlannerFn) -> Callable[[SQLAgentState], dict[str, Any]]:
    """Create the planning node using the supplied planner function."""

    def plan_node(state: SQLAgentState) -> dict[str, Any]:
        plan = planner(state)
        selected_targets = plan.selected_targets or [cast(str, target["name"]) for target in state.inspected_targets]
        if not plan.ready or not plan.sql:
            return {
                "status": "blocked",
                "plan": plan,
                "selected_targets": selected_targets,
                "rationale": plan.rationale,
                "last_error": plan.blocking_reason or "Planner could not produce a safe SQL query.",
                "trace": _append_trace(state, "planner blocked execution"),
            }

        return {
            "status": "planned",
            "plan": plan,
            "candidate_sql": plan.sql,
            "selected_targets": selected_targets,
            "rationale": plan.rationale,
            "trace": _append_trace(state, f"planned SQL for {', '.join(selected_targets)}"),
        }

    return plan_node


def execute_node(state: SQLAgentState) -> dict[str, Any]:
    """Execute the planned SQL."""
    if not state.candidate_sql:
        return {
            "status": "error",
            "last_error": "No SQL query was available for execution.",
            "trace": _append_trace(state, "execute skipped because no SQL was planned"),
        }

    result = run_sql(
        state.candidate_sql,
        database_path=state.database_path,
    )
    attempts = state.attempts + 1
    if result["status"] == "ok":
        return {
            "status": "complete",
            "attempts": attempts,
            "result": result,
            "trace": _append_trace(state, f"execute succeeded on attempt {attempts}"),
        }

    repair_target_names = [cast(str, suggestion["name"]) for suggestion in state.suggestions if suggestion.get("name")]
    repair_columns = {
        cast(str, target["name"]): [cast(str, column["name"]) for column in target.get("columns", []) if column.get("name")]
        for target in state.inspected_targets
        if target.get("name")
    }
    error_message = cast(str, result["message"])
    repair_hints = suggest_sql_error_repair(
        error_message,
        available_targets=repair_target_names,
        target_columns=repair_columns,
    )
    trace_message = f"execute failed on attempt {attempts}: {error_message}"
    if repair_hints:
        trace_message += f" ({len(repair_hints)} repair hints)"
    return {
        "status": "needs_repair",
        "attempts": attempts,
        "repair_hints": repair_hints,
        "result": result,
        "last_error": error_message,
        "trace": _append_trace(state, trace_message),
    }


def repair_node(state: SQLAgentState) -> dict[str, Any]:
    """Mark a repair pass before planning again."""
    return {
        "status": "repairing",
        "repair_count": state.repair_count + 1,
        "trace": _append_trace(state, f"repair pass {state.repair_count + 1}"),
    }


def _route_after_suggest(state: SQLAgentState) -> str:
    """Route after target suggestion."""
    if state.status != "suggested":
        return END
    if _needs_clarification(state.question):
        return "clarify"
    return "inspect"


def _route_after_inspect(state: SQLAgentState) -> str:
    """Route after target inspection."""
    return "plan" if state.status == "inspected" else END


def _route_after_plan(state: SQLAgentState) -> str:
    """Route after SQL planning."""
    return "execute" if state.status == "planned" else END


def _route_after_execute(state: SQLAgentState) -> str:
    """Route after execution."""
    if state.status == "complete":
        return END
    if state.status == "needs_repair" and state.repair_count < state.max_repairs:
        return "repair"
    return END


def create_sql_graph(planner: PlannerFn) -> CompiledStateGraph:
    """Create a minimal LangGraph SQL workflow."""
    builder = StateGraph(
        SQLAgentState,
        input_schema=SQLAgentInput,
        output_schema=SQLAgentOutput,
    )
    builder.add_node("suggest", suggest_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("inspect", inspect_node)
    builder.add_node("plan", make_plan_node(planner))
    builder.add_node("execute", execute_node)
    builder.add_node("repair", repair_node)

    builder.add_edge(START, "suggest")
    builder.add_conditional_edges(
        "suggest",
        _route_after_suggest,
        {
            "clarify": "clarify",
            "inspect": "inspect",
            END: END,
        },
    )
    builder.add_edge("clarify", END)
    builder.add_conditional_edges(
        "inspect",
        _route_after_inspect,
        {
            "plan": "plan",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "plan",
        _route_after_plan,
        {
            "execute": "execute",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "execute",
        _route_after_execute,
        {
            "repair": "repair",
            END: END,
        },
    )
    builder.add_edge("repair", "plan")
    return builder.compile()


class SQLAgent:
    """Minimal LangGraph SQL agent that orchestrates the standalone SQL tools."""

    def __init__(
        self,
        planner: PlannerFn | None = None,
        *,
        llm: BaseChatModel | None = None,
        model: str | None = None,
        temperature: float = 0,
        reasoning_effort: Literal["minimal", "low", "medium", "high"] = DEFAULT_SQL_AGENT_REASONING,
    ):
        self.planner = planner or make_llm_planner(
            llm,
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        self.graph = create_sql_graph(self.planner)

    def invoke(
        self,
        question: str,
        *,
        database_path: str | Path | None = None,
        max_suggestions: int = 3,
        max_repairs: int = 2,
        sample_rows: int = 3,
        text_value_hints: int = 3,
    ) -> SQLAgentOutput:
        """Run the SQL graph for one question."""
        result = self.graph.invoke(
            SQLAgentInput(
                question=question,
                database_path=None if database_path is None else str(Path(database_path).expanduser().resolve()),
                max_suggestions=max_suggestions,
                max_repairs=max_repairs,
                sample_rows=sample_rows,
                text_value_hints=text_value_hints,
            )
        )
        return SQLAgentOutput.model_validate(result)


def answer_sql_question(
    question: str,
    *,
    planner: PlannerFn | None = None,
    llm: BaseChatModel | None = None,
    model: str | None = None,
    database_path: str | Path | None = None,
    max_suggestions: int = 3,
    max_repairs: int = 2,
    sample_rows: int = 3,
    text_value_hints: int = 3,
) -> SQLAgentOutput:
    """Convenience wrapper for one-shot SQL agent execution."""
    agent = SQLAgent(
        planner=planner,
        llm=llm,
        model=model,
    )
    return agent.invoke(
        question,
        database_path=database_path,
        max_suggestions=max_suggestions,
        max_repairs=max_repairs,
        sample_rows=sample_rows,
        text_value_hints=text_value_hints,
    )
