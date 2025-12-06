from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import json
import time
import textwrap
import hashlib

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

from code import get_env_var
from .context_policies import ContextPayload, ContextPolicy
from .run_logging import AgentRunLogger, AgentStepLog
from .tools import Tool, ToolResult


@dataclass
class AgentStep:
    step_idx: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    tool_output: Optional[str]
    latency_sec: float
    usage: Dict[str, int]
    clarifying: bool = False
    clarifying_question: Optional[str] = None


@dataclass
class AgentRunResult:
    final_answer: str
    steps: List[AgentStep]
    metrics: Dict[str, Any]
    structured_output: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[str] = None
    iteration_entry: Dict[str, Any] = field(default_factory=dict)


class AgentRunner:
    """Simple ReAct-style agent runner parameterized by a context policy."""

    def __init__(
        self,
        *,
        policy: ContextPolicy,
        tools: Dict[str, Tool],
        logger: Optional[AgentRunLogger] = None,
        tracer: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.policy = policy
        self.tools = tools
        self.logger = logger
        self.tracer = tracer
        self.config = config or {}
        self.model = self.config.get("model", "gpt-4o-mini")
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_steps = int(self.config.get("max_steps", 4))
        self.result_schema = self.config.get("result_schema", "Return JSON with a 'proposal' object.")
        self.dataset_id = self.config.get("dataset_id", "nomad")
        self.agent_id = self.config.get("agent_id", self.model)

        api_key = get_env_var("OPENAI_API_KEY")
        self.use_stub = self.config.get("use_stub", False) or not api_key or not OPENAI_AVAILABLE
        self.client = OpenAI(api_key=api_key) if (not self.use_stub and OPENAI_AVAILABLE) else None

    # Public API -----------------------------------------------------------------
    def run(
        self,
        *,
        task_input: str,
        state: Dict[str, Any],
        run_id: str,
        task_id: str,
        reasoning_mode: str,
        iteration_idx: Optional[int] = None,
        log_run_detail: bool = True,
    ) -> AgentRunResult:
        context_payload = self.policy.build_context(task_input, state)
        messages = self._bootstrap_messages(context_payload, task_input, state)
        initial_messages = json.loads(json.dumps(messages))  # deep copy for logging
        steps: List[AgentStep] = []
        final_answer = ""
        structured_output: Dict[str, Any] = {}
        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        start = time.time()

        if self.logger:
            self.logger.start_run(
                run_id=run_id,
                task_id=task_id,
                policy_type=self.policy.name,
                reasoning_mode=reasoning_mode,
                user_input=task_input,
                metadata=context_payload.metadata,
            )

        for step_idx in range(1, self.max_steps + 1):
            span_cm = (
                self.tracer.span(
                    "agent.runner.step",
                    {
                        "agent.step_idx": step_idx,
                        "agent.policy": self.policy.name,
                        "agent.reasoning_mode": reasoning_mode,
                    },
                )
                if self.tracer
                else nullcontext()
            )
            with span_cm as step_span:
                if step_span:
                    prompt_summary = _truncate_json(messages, 900)
                    self.tracer.set_attributes(
                        step_span,
                        {
                            "agent.prompt.summary": prompt_summary,
                            "agent.prompt.hash": _hash_text(prompt_summary),
                            "agent.prompt.full": json.dumps(messages),
                        },
                    )
                response_text, usage = self._call_model(messages)
                total_usage = _accumulate_usage(total_usage, usage)
                command = self._parse_action(response_text)
                thought = command.get("thought", "").strip()
                action = command.get("action", "").strip()
                action_input = command.get("action_input") or {}
                clarifying = action == "ask_clarifying_question"
                clarifying_question = action_input.get("question") if clarifying else None

                observation = ""
                tool_output = None
                if action == "final_answer":
                    final_answer = action_input.get("answer", response_text)
                    structured_output = action_input
                    observation = "Final answer produced."
                    messages.append({"role": "assistant", "content": response_text})
                elif action in self.tools:
                    tool = self.tools[action]
                    tool_result = tool.run(action_input, state)
                    tool_output = tool_result.content
                    observation = f"{tool.name} => {tool_output}"
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Observation: {tool_result.content}",
                        }
                    )
                else:
                    observation = "Unknown action requested; stopping."
                    final_answer = response_text

                step_latency = usage.get("latency_sec", 0.0)
                agent_step = AgentStep(
                    step_idx=step_idx,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation,
                    tool_output=tool_output,
                    latency_sec=step_latency,
                    usage=usage,
                    clarifying=clarifying,
                    clarifying_question=clarifying_question,
                )
                steps.append(agent_step)

                if self.logger:
                    self.logger.log_step(
                        AgentStepLog(
                            step_idx=step_idx,
                            thought=thought,
                            action=action,
                            action_input=action_input,
                            observation=observation,
                            tool_output=tool_output,
                            latency_sec=step_latency,
                            tokens=usage,
                            clarifying=clarifying,
                            clarifying_question=clarifying_question,
                        )
                    )

                if step_span and usage:
                    self.tracer.set_attributes(
                        step_span,
                        {
                            "agent.action": action,
                            "agent.thought": thought,
                            "agent.observation": observation,
                            "agent.latency": step_latency,
                            "agent.usage.total_tokens": usage.get("total_tokens", 0),
                            "agent.usage.input_tokens": usage.get("input_tokens", 0),
                            "agent.usage.output_tokens": usage.get("output_tokens", 0),
                        },
                    )

                if action == "final_answer" or not action or observation.startswith("Unknown action"):
                    break

        elapsed = time.time() - start
        metrics = {
            "policy": self.policy.name,
            "steps": len(steps),
            "clarifying_questions": sum(1 for s in steps if s.clarifying),
            "total_tokens": total_usage.get("total_tokens", 0),
            "latency_sec": elapsed,
        }

        if self.logger:
            self.logger.end_run(
                final_answer=final_answer,
                metrics=metrics,
                structured_output=structured_output,
            )

        iteration_entry = _build_iteration_entry(
            iteration_idx=iteration_idx,
            task_input=task_input,
            state=state,
            initial_messages=initial_messages,
            final_answer=final_answer,
            metrics=metrics,
            structured_output=structured_output,
            steps=steps,
        )

        if log_run_detail:
            _append_iteration_trace(
                run_id=run_id,
                task_id=task_id,
                policy=self.policy.name,
                reasoning_mode=reasoning_mode,
                iteration_entry=iteration_entry,
                dataset_id=self.dataset_id,
                agent_id=self.agent_id,
        )

        return AgentRunResult(
            final_answer=final_answer,
            steps=steps,
            metrics=metrics,
            structured_output=structured_output,
            raw_response=final_answer,
            iteration_entry=iteration_entry,
        )

    # Internal helpers ----------------------------------------------------------
    def _bootstrap_messages(
        self,
        payload: ContextPayload,
        task_input: str,
        state: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        tools_description = self._tool_instructions()
        memory_hint = state.get("memory_hint", "")
        history = state.get("history") or []
        history_text = json.dumps(history, indent=2) if history else "[]"

        user_prompt = textwrap.dedent(
            f"""
            ## Task
            {task_input}

            ## Known context (truncated)
            {state.get('context_excerpt', 'N/A')}

            ## Recent iterations (JSON)
            {history_text}

            ## Result schema
            {self.result_schema}
            """
        ).strip()

        messages = [
            {"role": "system", "content": payload.system_prompt},
            {"role": "system", "content": payload.action_schema_hint},
            {"role": "system", "content": payload.clarifier_instructions},
            {"role": "system", "content": tools_description},
        ]
        if memory_hint:
            messages.append({"role": "system", "content": memory_hint})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _tool_instructions(self) -> str:
        lines = ["Available tool catalog:"]
        for tool in self.tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def _call_model(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
        if self.use_stub or not self.client:
            return self._stub_completion(messages)

        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=int(self.config.get("max_tokens", 300)),
            messages=messages,
        )
        latency = time.time() - t0
        text = resp.choices[0].message.content or ""
        usage = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
            "latency_sec": latency,
        }
        return text, usage

    def _stub_completion(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
        """Deterministic fallback when no LLM is available."""
        # Simple heuristic: cycle through retrieve -> summarize -> final_answer
        existing_actions = sum(1 for m in messages if m.get("role") == "assistant")
        if existing_actions < 1:
            response = json.dumps(
                {
                    "thought": "Need the latest dataset context to reason about hyperparameters.",
                    "action": "retrieve_docs",
                    "action_input": {"k": 2},
                }
            )
        elif existing_actions < 2:
            response = json.dumps(
                {
                    "thought": "Summarize the retrieved chunks to focus on important signals.",
                    "action": "summarize_chunks",
                    "action_input": {"text": " ".join(m["content"] for m in messages if m["role"] == "user")},
                }
            )
        else:
            response = json.dumps(
                {
                    "thought": "Provide a conservative configuration proposal.",
                    "action": "final_answer",
                    "action_input": {
                        "answer": "Stub proposal generated without LLM.",
                        "proposal": {
                            "learning_rate": 0.1,
                            "max_depth": 8,
                            "max_iter": 250,
                            "l2_regularization": 0.1,
                            "max_leaf_nodes": 64,
                            "min_samples_leaf": 20,
                        },
                    },
                }
            )
        usage = {
            "input_tokens": len(str(messages)) // 4,
            "output_tokens": len(response) // 4,
            "total_tokens": (len(str(messages)) + len(response)) // 4,
            "latency_sec": 0.01,
        }
        return response, usage

    def _parse_action(self, response_text: str) -> Dict[str, Any]:
        text = response_text.strip()
        if "```" in text:
            # attempt to extract JSON block
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("{") and part.endswith("}"):
                    text = part
                    break
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # fallback schema
            return {
                "thought": text,
                "action": "final_answer",
                "action_input": {"answer": text},
            }


def _accumulate_usage(total: Dict[str, int], usage: Dict[str, int]) -> Dict[str, int]:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + usage.get(key, 0)
    return total


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _truncate_json(obj: Any, limit: int) -> str:
    try:
        text = json.dumps(obj)
    except Exception:
        text = str(obj)
    return _truncate_text(text, limit)


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _build_iteration_entry(
    *,
    iteration_idx: Optional[int],
    task_input: str,
    state: Dict[str, Any],
    initial_messages: List[Dict[str, Any]],
    final_answer: str,
    metrics: Dict[str, Any],
    structured_output: Dict[str, Any],
    steps: List[AgentStep],
) -> Dict[str, Any]:
    return {
        "iteration_idx": iteration_idx,
        "task_input": task_input,
        "state": state,
        "initial_messages": initial_messages,
        "final_answer": final_answer,
        "metrics": metrics,
        "structured_output": structured_output,
        "steps": [
            {
                "step_idx": s.step_idx,
                "thought": s.thought,
                "action": s.action,
                "action_input": s.action_input,
                "observation": s.observation,
                "tool_output": s.tool_output,
                "latency_sec": s.latency_sec,
                "usage": s.usage,
                "clarifying": s.clarifying,
                "clarifying_question": s.clarifying_question,
            }
            for s in steps
        ],
        "clarifying_questions": [
            {
                "step_idx": s.step_idx,
                "question": s.clarifying_question,
                "action_input": s.action_input,
                "observation": s.observation,
                "answer": s.tool_output,
            }
            for s in steps
            if s.clarifying
        ],
    }


def _append_iteration_trace(
    *,
    run_id: str,
    task_id: str,
    policy: str,
    reasoning_mode: str,
    iteration_entry: Dict[str, Any],
    dataset_id: str,
    agent_id: str,
    log_root: Path | None = None,
) -> None:
    """Append an iteration event to traces/{task_id}/{run_id}.jsonl."""
    root = log_root or Path("traces") / task_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{run_id}.jsonl"

    event = {
        "run_id": run_id,
        "event_type": "agent.iteration",
        "step_idx": iteration_entry.get("iteration_idx"),
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task_id": task_id,
        "dataset_id": dataset_id,
        "agent_id": agent_id,
        "details": {
            "policy_type": policy,
            "reasoning_mode": reasoning_mode,
            "iteration": iteration_entry,
        },
    }

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")

