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


@dataclass
class PolicyConfig:
    """Configuration for a context policy."""

    name: str
    max_retrieved_chunks: int
    chunk_char_limit: int
    summary_char_limit: int
    priority: str
    token_budget_hint: str
    system_prompt_template: str
    action_hint_template: str
    clarifier_text: str


SHORT_CONTEXT_PRESET = PolicyConfig(
    name="short_context",
    max_retrieved_chunks=3,
    chunk_char_limit=600,
    summary_char_limit=400,
    priority="precision",
    token_budget_hint="low",
    system_prompt_template=textwrap.dedent(
        """
        You are an ML experimentation agent following the SHORT context policy.
        Only the most relevant snippets are available to you. Minimize token usage,
        aggressively detect ambiguity, and ask clarifying questions whenever the task
        request is underspecified. Never invent data that is not explicitly retrieved.
        Task input: {task_input}
        """
    ).strip(),
    action_hint_template=textwrap.dedent(
        """
        Available actions:
          - retrieve_docs: fetch up to k focused context snippets (cheap but limited)
          - summarize_chunks: compress retrieved snippets before continuing
          - ask_clarifying_question: ask the user/system for missing task details
          - final_answer: present the final answer or proposal as JSON
        Always respond with JSON: {"thought": "...", "action": "<name>", "action_input": {...}}
        """
    ).strip(),
    clarifier_text=(
        "Use ask_clarifying_question when critical configuration details, targets, or "
        "constraints are missing. Do not proceed to final_answer until ambiguities are resolved "
        "or you explicitly state the remaining uncertainty."
    ),
)

LONG_CONTEXT_PRESET = PolicyConfig(
    name="long_context",
    max_retrieved_chunks=12,
    chunk_char_limit=1200,
    summary_char_limit=800,
    priority="recall",
    token_budget_hint="high",
    system_prompt_template=textwrap.dedent(
        """
        You are an ML experimentation agent following the LONG context policy.
        You may access large portions of the dataset context, previous iterations,
        and intermediate artifacts all at once. Reason across the full context to
        avoid missing important signals. Prefer single-pass answers when feasible.
        Task input: {task_input}
        """
    ).strip(),
    action_hint_template=textwrap.dedent(
        """
        Available actions:
          - retrieve_docs: fetch many context snippets (expensive but comprehensive)
          - summarize_chunks: build high-level summaries to avoid distraction
          - ask_clarifying_question: only if absolutely necessary
          - final_answer: produce the best proposal/answer with supporting rationale
        Respond strictly in JSON: {"thought": "...", "action": "<name>", "action_input": {...}}
        """
    ).strip(),
    clarifier_text=(
        "Only ask clarifying questions if you cannot proceed after reviewing all provided "
        "context. Most tasks should complete without extra user turns."
    ),
)


class ContextPolicy:
    """Base class for all context policies."""

    name: str

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    def build_context(self, task_input: str, state: Dict[str, Any]) -> ContextPayload:
        raise NotImplementedError


class ConfigurableContextPolicy(ContextPolicy):
    """Unified context policy with configurable presets."""

    def __init__(self, preset: PolicyConfig, config_overrides: Optional[Dict[str, Any]] = None):
        self.preset = preset
        merged = {
            "max_retrieved_chunks": preset.max_retrieved_chunks,
            "chunk_char_limit": preset.chunk_char_limit,
            "summary_char_limit": preset.summary_char_limit,
        }
        if config_overrides:
            merged.update(config_overrides)
        super().__init__(preset.name, merged)

    def build_context(self, task_input: str, state: Dict[str, Any]) -> ContextPayload:
        system_prompt = self.preset.system_prompt_template.format(task_input=task_input)

        retrieval_config = {
            "max_retrieved_chunks": self.config["max_retrieved_chunks"],
            "chunk_char_limit": self.config["chunk_char_limit"],
            "summary_char_limit": self.config["summary_char_limit"],
        }

        metadata = {
            "policy": self.name,
            "priority": self.preset.priority,
            "token_budget_hint": self.preset.token_budget_hint,
        }

        return ContextPayload(
            policy_name=self.name,
            system_prompt=system_prompt,
            action_schema_hint=self.preset.action_hint_template,
            retrieval_config=retrieval_config,
            clarifier_instructions=self.preset.clarifier_text,
            metadata=metadata,
        )


class ShortContextPolicy(ConfigurableContextPolicy):
    """Restrictive policy that encourages clarification before answering."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(SHORT_CONTEXT_PRESET, config)


class LongContextPolicy(ConfigurableContextPolicy):
    """Policy that favors rich context and fewer clarification turns."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(LONG_CONTEXT_PRESET, config)


def create_policy(policy_type: str, config_map: Optional[Dict[str, Any]] = None) -> ContextPolicy:
    policy_type = (policy_type or "short_context").lower()
    config_map = config_map or {}

    if policy_type == "long_context":
        return LongContextPolicy(config_map)
    return ShortContextPolicy(config_map)
