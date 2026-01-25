"""Tool definitions for agent systems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolResult:
    content: str
    metadata: Dict[str, Any]


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[[Dict[str, Any], Dict[str, Any]], ToolResult]

    def run(self, action_input: Dict[str, Any], state: Dict[str, Any]) -> ToolResult:
        return self.handler(action_input, state)


# Tool description constants
NOMAD_TOOL_DESCRIPTIONS = {
    "retrieve_docs": "Fetches focused snippets from the NOMAD dataset context.",
    "summarize_chunks": "Summarizes provided text to highlight the most important signals.",
    "ask_clarifying_question": "Asks for missing task details; returns stored defaults when available.",
}

TOY_TOOL_DESCRIPTIONS = {
    "retrieve_docs": "Fetches context about the Toy benchmark task and parameters.",
    "summarize_chunks": "Summarizes text to highlight important information.",
    "ask_clarifying_question": "Asks for clarification about the task.",
}


def _serialize_context_chunks(context_summary: Dict[str, Any]) -> List[str]:
    chunks: List[str] = []
    for key, value in context_summary.items():
        if isinstance(value, dict):
            inner = ", ".join(f"{k}: {v}" for k, v in value.items())
            chunks.append(f"{key}: {inner}")
        elif isinstance(value, list):
            preview = ", ".join(str(item) for item in value[:5])
            chunks.append(f"{key} (first items): {preview}")
        else:
            chunks.append(f"{key}: {value}")
    return chunks


def _create_retrieve_handler(
    chunks: List[str], retrieval_config: Dict[str, Any]
) -> Callable[[Dict[str, Any], Dict[str, Any]], ToolResult]:
    """Create retrieve_docs handler with given chunks and config."""

    def _retrieve_docs(action_input: Dict[str, Any], state: Dict[str, Any]) -> ToolResult:
        k = int(action_input.get("k") or retrieval_config.get("max_retrieved_chunks", 3))
        k = max(1, min(k, len(chunks)))
        snippet = "\n".join(chunks[:k])
        return ToolResult(
            content=snippet[: retrieval_config.get("chunk_char_limit", 600)],
            metadata={"chunks_returned": k},
        )

    return _retrieve_docs


def _create_summarize_handler(
    retrieval_config: Dict[str, Any]
) -> Callable[[Dict[str, Any], Dict[str, Any]], ToolResult]:
    """Create summarize handler with given config."""

    def _summarize(action_input: Dict[str, Any], state: Dict[str, Any]) -> ToolResult:
        text = action_input.get("text") or ""
        limit = retrieval_config.get("summary_char_limit", 400)
        summary_text = " ".join(text.strip().split())[:limit]
        return ToolResult(
            content=summary_text,
            metadata={"original_length": len(text), "summary_length": len(summary_text)},
        )

    return _summarize


def _create_clarify_handler(
    clarifier_defaults: Dict[str, str], normalize_questions: bool
) -> Callable[[Dict[str, Any], Dict[str, Any]], ToolResult]:
    """Create clarify handler with defaults and normalization setting."""

    def _clarify(action_input: Dict[str, Any], state: Dict[str, Any]) -> ToolResult:
        question = (action_input.get("question") or "").strip()
        lookup_key = question.lower()
        answer = clarifier_defaults.get(lookup_key)
        if not answer:
            hint_fields = state.get("clarification_hints", {})
            answer = hint_fields.get(lookup_key, "No additional information available.")
        # Store normalized or original question based on setting
        stored_question = lookup_key if normalize_questions else question
        return ToolResult(
            content=answer,
            metadata={"question": stored_question, "auto_answered": True},
        )

    return _clarify


def build_tools(
    *,
    context_summary: Dict[str, Any],
    retrieval_config: Dict[str, Any],
    clarifier_defaults: Optional[Dict[str, str]] = None,
    tool_descriptions: Optional[Dict[str, str]] = None,
    normalize_questions: bool = True,
) -> Dict[str, Tool]:
    """Generic tool factory for all benchmarks.

    Args:
        context_summary: Dataset/task context to serialize into chunks.
        retrieval_config: Config with max_retrieved_chunks, chunk_char_limit, etc.
        clarifier_defaults: Default answers for clarifying questions.
        tool_descriptions: Override default tool descriptions per benchmark.
        normalize_questions: Whether to lowercase questions in clarify metadata.

    Returns:
        Dictionary of tool name to Tool instance.
    """
    chunks = _serialize_context_chunks(context_summary)
    clarifier_defaults = clarifier_defaults or {}
    descriptions = tool_descriptions or {}

    return {
        "retrieve_docs": Tool(
            name="retrieve_docs",
            description=descriptions.get("retrieve_docs", "Retrieves relevant documentation chunks."),
            handler=_create_retrieve_handler(chunks, retrieval_config),
        ),
        "summarize_chunks": Tool(
            name="summarize_chunks",
            description=descriptions.get("summarize_chunks", "Summarizes provided text."),
            handler=_create_summarize_handler(retrieval_config),
        ),
        "ask_clarifying_question": Tool(
            name="ask_clarifying_question",
            description=descriptions.get("ask_clarifying_question", "Asks for clarification."),
            handler=_create_clarify_handler(clarifier_defaults, normalize_questions),
        ),
    }


# Backward-compatible wrapper functions
def build_nomad_tools(
    *,
    context_summary: Dict[str, Any],
    retrieval_config: Dict[str, Any],
    clarifier_defaults: Optional[Dict[str, str]] = None,
) -> Dict[str, Tool]:
    """Build NOMAD-specific tools. Wrapper around build_tools()."""
    return build_tools(
        context_summary=context_summary,
        retrieval_config=retrieval_config,
        clarifier_defaults=clarifier_defaults,
        tool_descriptions=NOMAD_TOOL_DESCRIPTIONS,
        normalize_questions=False,
    )


def build_toy_tools(
    *,
    context_summary: Dict[str, Any],
    retrieval_config: Dict[str, Any],
    clarifier_defaults: Optional[Dict[str, str]] = None,
) -> Dict[str, Tool]:
    """Build Toy-specific tools. Wrapper around build_tools()."""
    defaults = clarifier_defaults or {
        "what metric should i optimize?": "Accuracy - higher is better.",
        "what are the parameter bounds?": "C: 0.01-100.0, max_iter: 10-1000",
    }
    return build_tools(
        context_summary=context_summary,
        retrieval_config=retrieval_config,
        clarifier_defaults=defaults,
        tool_descriptions=TOY_TOOL_DESCRIPTIONS,
        normalize_questions=True,
    )
