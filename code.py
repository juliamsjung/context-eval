"""
Toy LLM system using the OpenAI Python SDK and a GPT-5 key.

- One-shot call via the Responses API.
- Deterministic-ish defaults (temperature=0).
- Minimal metrics so you can log basic run info.

Requires:
  pip install openai
  export OPENAI_API_KEY=...
"""

from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path
import json
import os
import time

from openai import OpenAI


@dataclass
class Result:
    success: bool
    output: str
    metrics: Dict[str, Any]


def load_config(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _build_prompt(system_prompt: str, user_input: str) -> str:
    # Simple, explicit delimiting; easy to inspect in traces.
    return f"<SYSTEM>\n{system_prompt}\n</SYSTEM>\n<USER>\n{user_input}\n</USER>"


def run(config: Dict[str, Any]) -> Result:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export your key before running."
        )

    model = config.get("model", "gpt-5")
    system_prompt = config.get("system_prompt", "You are a precise, concise assistant.")
    task = config.get("task", {})
    user_input = task.get("input", "Say hello in one sentence.")
    success_contains = [s.lower() for s in task.get("success_contains", [])]

    temperature = float(config.get("temperature", 0.0))
    max_out = int(config.get("max_output_tokens", 256))

    prompt = _build_prompt(system_prompt, user_input)

    client = OpenAI()
    t0 = time.time()
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_out,
    )
    elapsed = time.time() - t0

    # Prefer SDK convenience; fall back if absent.
    text = getattr(resp, "output_text", None)
    if text is None:
        # Very conservative fallback in case SDK changes:
        try:
            # Many SDKs expose the first text chunk like this:
            text = resp.output[0].content[0].text  # type: ignore[attr-defined]
        except Exception:
            text = str(resp)

    usage = getattr(resp, "usage", {}) or {}
    input_tokens = getattr(usage, "input_tokens", None) or usage.get("input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", None) or usage.get("output_tokens", 0)
    total_tokens = input_tokens + output_tokens

    # Minimal success heuristic (optional)
    success = True
    if success_contains:
        lower = text.lower()
        success = all(s in lower for s in success_contains)

    metrics = {
        "provider": "openai",
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_out,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed, 4),
    }

    return Result(success=success, output=text, metrics=metrics)