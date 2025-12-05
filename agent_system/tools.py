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


def build_nomad_tools(
    *,
    context_summary: Dict[str, Any],
    retrieval_config: Dict[str, Any],
    clarifier_defaults: Optional[Dict[str, str]] = None,
) -> Dict[str, Tool]:
    """Create a toolset tailored for the NOMAD benchmark."""

    chunks = _serialize_context_chunks(context_summary)
    clarifier_defaults = clarifier_defaults or {}

    def _retrieve_docs(action_input: Dict[str, Any], state: Dict[str, Any]) -> ToolResult:
        k = int(action_input.get("k") or retrieval_config.get("max_retrieved_chunks", 3))
        k = max(1, min(k, len(chunks)))
        snippet = "\n".join(chunks[:k])
        return ToolResult(
            content=snippet[: retrieval_config.get("chunk_char_limit", 600)],
            metadata={"chunks_returned": k},
        )

    def _summarize(action_input: Dict[str, Any], state: Dict[str, Any]) -> ToolResult:
        text = action_input.get("text") or ""
        limit = retrieval_config.get("summary_char_limit", 400)
        summary = text.strip().split("\n")
        summary_text = " ".join(line.strip() for line in summary)[:limit]
        return ToolResult(
            content=summary_text,
            metadata={"original_length": len(text), "summary_length": len(summary_text)},
        )

    def _clarify(action_input: Dict[str, Any], state: Dict[str, Any]) -> ToolResult:
        question = (action_input.get("question") or "").strip()
        answer = clarifier_defaults.get(question.lower())
        if not answer:
            # Try to infer answer from state hints
            hint_fields = state.get("clarification_hints", {})
            answer = hint_fields.get(question.lower(), "No additional information available.")
        return ToolResult(
            content=answer,
            metadata={"question": question, "auto_answered": True},
        )

    return {
        "retrieve_docs": Tool(
            name="retrieve_docs",
            description="Fetches focused snippets from the NOMAD dataset context.",
            handler=_retrieve_docs,
        ),
        "summarize_chunks": Tool(
            name="summarize_chunks",
            description="Summarizes provided text to highlight the most important signals.",
            handler=_summarize,
        ),
        "ask_clarifying_question": Tool(
            name="ask_clarifying_question",
            description="Asks for missing task details; returns stored defaults when available.",
            handler=_clarify,
        ),
    }

