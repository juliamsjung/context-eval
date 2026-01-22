"""Context policies for agent behavior control."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import textwrap


@dataclass
class ContextPayload:
    """Data returned by a ContextPolicy for the agent runner."""

    policy_name: str
    system_prompt: str
    action_schema_hint: str
    retrieval_config: Dict[str, Any] = field(default_factory=dict)
    clarifier_instructions: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextPolicy:
    """Base class for all context policies."""

    name: str

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    def build_context(self, task_input: str, state: Dict[str, Any]) -> ContextPayload:
        raise NotImplementedError


class ShortContextPolicy(ContextPolicy):
    """Restrictive policy that encourages clarification before answering."""

    DEFAULTS = {
        "max_retrieved_chunks": 3,
        "chunk_char_limit": 600,
        "summary_char_limit": 400,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        merged = {**ShortContextPolicy.DEFAULTS, **(config or {})}
        super().__init__("short_context", merged)

    def build_context(self, task_input: str, state: Dict[str, Any]) -> ContextPayload:
        system_prompt = textwrap.dedent(
            f"""
            You are an ML experimentation agent following the SHORT context policy.
            Only the most relevant snippets are available to you. Minimize token usage,
            aggressively detect ambiguity, and ask clarifying questions whenever the task
            request is underspecified. Never invent data that is not explicitly retrieved.
            Task input: {task_input}
            """
        ).strip()

        action_hint = textwrap.dedent(
            """
            Available actions:
              - retrieve_docs: fetch up to k focused context snippets (cheap but limited)
              - summarize_chunks: compress retrieved snippets before continuing
              - ask_clarifying_question: ask the user/system for missing task details
              - final_answer: present the final answer or proposal as JSON
            Always respond with JSON: {"thought": "...", "action": "<name>", "action_input": {...}}
            """
        ).strip()

        clarifier = (
            "Use ask_clarifying_question when critical configuration details, targets, or "
            "constraints are missing. Do not proceed to final_answer until ambiguities are resolved "
            "or you explicitly state the remaining uncertainty."
        )

        retrieval_config = {
            "max_retrieved_chunks": self.config["max_retrieved_chunks"],
            "chunk_char_limit": self.config["chunk_char_limit"],
            "summary_char_limit": self.config["summary_char_limit"],
        }

        metadata = {
            "policy": self.name,
            "priority": "precision",
            "token_budget_hint": "low",
        }

        return ContextPayload(
            policy_name=self.name,
            system_prompt=system_prompt,
            action_schema_hint=action_hint,
            retrieval_config=retrieval_config,
            clarifier_instructions=clarifier,
            metadata=metadata,
        )


class LongContextPolicy(ContextPolicy):
    """Policy that favors rich context and fewer clarification turns."""

    DEFAULTS = {
        "max_retrieved_chunks": 12,
        "chunk_char_limit": 1200,
        "summary_char_limit": 800,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        merged = {**LongContextPolicy.DEFAULTS, **(config or {})}
        super().__init__("long_context", merged)

    def build_context(self, task_input: str, state: Dict[str, Any]) -> ContextPayload:
        system_prompt = textwrap.dedent(
            f"""
            You are an ML experimentation agent following the LONG context policy.
            You may access large portions of the dataset context, previous iterations,
            and intermediate artifacts all at once. Reason across the full context to
            avoid missing important signals. Prefer single-pass answers when feasible.
            Task input: {task_input}
            """
        ).strip()

        action_hint = textwrap.dedent(
            """
            Available actions:
              - retrieve_docs: fetch many context snippets (expensive but comprehensive)
              - summarize_chunks: build high-level summaries to avoid distraction
              - ask_clarifying_question: only if absolutely necessary
              - final_answer: produce the best proposal/answer with supporting rationale
            Respond strictly in JSON: {"thought": "...", "action": "<name>", "action_input": {...}}
            """
        ).strip()

        clarifier = (
            "Only ask clarifying questions if you cannot proceed after reviewing all provided "
            "context. Most tasks should complete without extra user turns."
        )

        retrieval_config = {
            "max_retrieved_chunks": self.config["max_retrieved_chunks"],
            "chunk_char_limit": self.config["chunk_char_limit"],
            "summary_char_limit": self.config["summary_char_limit"],
        }

        metadata = {
            "policy": self.name,
            "priority": "recall",
            "token_budget_hint": "high",
        }

        return ContextPayload(
            policy_name=self.name,
            system_prompt=system_prompt,
            action_schema_hint=action_hint,
            retrieval_config=retrieval_config,
            clarifier_instructions=clarifier,
            metadata=metadata,
        )


def create_policy(policy_type: str, config_map: Optional[Dict[str, Any]] = None) -> ContextPolicy:
    policy_type = (policy_type or "short_context").lower()
    config_map = config_map or {}

    if policy_type == "long_context":
        return LongContextPolicy(config_map)
    return ShortContextPolicy(config_map)
