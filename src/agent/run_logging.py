"""Agent run logging utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


@dataclass
class AgentStepLog:
    step_idx: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: Optional[str]
    tool_output: Optional[str]
    latency_sec: float
    tokens: Dict[str, int]
    clarifying: bool = False
    clarifying_question: Optional[str] = None


class AgentRunLogger:
    """Append-only JSONL logger for agent runs."""

    def __init__(self, log_path: str | Path = "logs/agent_runs.jsonl"):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_record: Dict[str, Any] = {}
        self.step_records: List[AgentStepLog] = []

    def start_run(
        self,
        *,
        run_id: str,
        task_id: str,
        policy_type: str,
        reasoning_mode: str,
        user_input: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_record = {
            "run_id": run_id,
            "task_id": task_id,
            "policy_type": policy_type,
            "reasoning_mode": reasoning_mode,
            "user_input": user_input,
            "metadata": metadata or {},
            "timestamp": _now_iso(),
        }
        self.step_records = []

    def log_step(self, step: AgentStepLog) -> None:
        self.step_records.append(step)

    def end_run(
        self,
        *,
        final_answer: str,
        metrics: Dict[str, Any],
        structured_output: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            **self.run_record,
            "final_answer": final_answer,
            "metrics": metrics,
            "structured_output": structured_output or {},
            "steps": [asdict(step) for step in self.step_records],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
