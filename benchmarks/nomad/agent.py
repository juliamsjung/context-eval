from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

from code import get_env_var
from benchmarks.nomad.env import NomadEnv

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_WINDOW = 5
MAX_CONTEXT_ATTR_LENGTH = 512

PARAM_BOUNDS = {
    "learning_rate": (0.01, 0.5),
    "max_depth": (2, 16),
    "max_iter": (50, 1000),
    "l2_regularization": (0.0, 2.0),
    "max_leaf_nodes": (15, 255),
    "min_samples_leaf": (5, 200),
}


def _record_history_entry(
    history: List[Dict[str, Any]],
    *,
    step: int,
    config: Dict[str, Any],
    results: Dict[str, Any],
    proposal_source: str,
    context_summary: Optional[Dict[str, Any]] = None,
) -> None:
    history.append(
        {
            "step": step,
            "config": config.copy(),
            "results": results.copy(),
            "proposal_source": proposal_source,
            "context": context_summary or {},
        }
    )


def _history_window(history: Sequence[Dict[str, Any]], window: int) -> List[Dict[str, Any]]:
    recent: Sequence[Dict[str, Any]] = history[-window:]
    payload: List[Dict[str, Any]] = []
    for entry in recent:
        payload.append(
            {
                "step": entry.get("step"),
                "proposal_source": entry.get("proposal_source"),
                "config": entry.get("config"),
                "results": entry.get("results"),
            }
        )
    return payload


def _context_str(summary: Optional[Dict[str, Any]]) -> Optional[str]:
    if not summary:
        return None
    blob = json.dumps(summary, sort_keys=True)
    if len(blob) <= MAX_CONTEXT_ATTR_LENGTH:
        return blob
    return blob[: MAX_CONTEXT_ATTR_LENGTH - 3] + "..."


def _safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return max(low, min(high, value))


def _sanitize_proposal(proposal: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, (low, high) in PARAM_BOUNDS.items():
        if key not in proposal:
            continue
        val = float(proposal[key])
        if key in {"max_depth", "max_leaf_nodes", "max_iter", "min_samples_leaf"}:
            sanitized[key] = int(_clamp(round(val), (low, high)))
        else:
            sanitized[key] = _clamp(val, (low, high))
    return sanitized


def _propose_config(
    config: Dict[str, Any],
    results: Dict[str, Any],
    history: List[Dict[str, Any]],
    *,
    context_summary: Optional[Dict[str, Any]] = None,
    history_window: int = DEFAULT_HISTORY_WINDOW,
) -> Optional[Dict[str, Any]]:
    if not OPENAI_AVAILABLE:
        return None

    api_key = get_env_var("OPENAI_API_KEY")
    if not api_key:
        return None

    prompt_payload = {
        "current_config": {k: config[k] for k in PARAM_BOUNDS.keys() if k in config},
        "latest_results": results,
        "recent_history": _history_window(history, history_window),
    }
    if context_summary:
        prompt_payload["dataset_context"] = context_summary

    user_prompt = (
        "You are tuning a HistGradientBoostingRegressor to predict bandgap energy (eV) "
        "from the NOMAD 2018 dataset. Use the structured information below to recommend "
        "a new configuration that lowers the evaluation metric (default: MAE).\n"
        f"{json.dumps(prompt_payload, indent=2)}\n\n"
        "Return JSON with numeric keys among "
        f"{list(PARAM_BOUNDS.keys())}. Keep values within reasonable ranges."
    )

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=250,
        messages=[
            {
                "role": "system",
                "content": "You are an ML assistant optimizing gradient boosting hyperparameters.",
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content or ""
    try:
        proposal = _safe_parse_json(content)
    except Exception:
        logger.warning("LLM response was not valid JSON: %s", content)
        return None

    sanitized = _sanitize_proposal(proposal)
    return sanitized or None


def _fallback_config(current: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
    factor = 1.15 if step_idx % 2 == 0 else 0.85
    updated = {
        "learning_rate": _clamp(current["learning_rate"] * factor, PARAM_BOUNDS["learning_rate"]),
        "max_depth": int(
            _clamp(
                current["max_depth"] + (1 if step_idx % 2 == 0 else -1),
                PARAM_BOUNDS["max_depth"],
            )
        ),
        "max_iter": int(
            _clamp(
                current["max_iter"] + (50 if step_idx % 2 == 0 else -30),
                PARAM_BOUNDS["max_iter"],
            )
        ),
        "l2_regularization": _clamp(
            current["l2_regularization"] * (1.3 if step_idx % 2 == 0 else 0.7),
            PARAM_BOUNDS["l2_regularization"],
        ),
        "max_leaf_nodes": int(
            _clamp(
                current["max_leaf_nodes"] + (8 if step_idx % 2 == 0 else -8),
                PARAM_BOUNDS["max_leaf_nodes"],
            )
        ),
        "min_samples_leaf": int(
            _clamp(
                current["min_samples_leaf"] + (-2 if step_idx % 2 == 0 else 2),
                PARAM_BOUNDS["min_samples_leaf"],
            )
        ),
    }
    return updated


def run_nomad_bench(
    num_steps: int = 3,
    tracer: Any = None,
    *,
    history_window: int = DEFAULT_HISTORY_WINDOW,
) -> Dict[str, Any]:
    env = NomadEnv()
    config = env.read_config()
    context_summary = env.read_context()
    history: List[Dict[str, Any]] = []

    print("===> Baseline run (NOMAD)")
    baseline_results = env.run_train()
    _record_history_entry(
        history,
        step=0,
        config=config,
        results=baseline_results,
        proposal_source="baseline",
        context_summary=context_summary,
    )
    last_results = baseline_results

    for step in range(1, num_steps + 1):
        print(f"\n===> NOMAD Step {step}/{num_steps}")
        proposal = _propose_config(
            config,
            last_results,
            history,
            context_summary=context_summary,
            history_window=history_window,
        )
        proposal_source = "llm"
        if not proposal:
            proposal = _fallback_config(config, step)
            proposal_source = "heuristic"
            print("Using heuristic proposal:", proposal)
        else:
            print("LLM proposal:", proposal)

        config.update(proposal)
        env.write_config(config)

        span_attributes = {
            "step": step,
            "num_steps": num_steps,
            "proposal_source": proposal_source,
            **{f"config.{k}": config.get(k) for k in PARAM_BOUNDS.keys()},
        }
        if history:
            prev = history[-1]
            latest_metric = prev.get("results", {}).get("metric_value")
            span_attributes.update(
                {
                    "prev.step": prev.get("step"),
                    "prev.metric": latest_metric,
                    "prev.metric_name": prev.get("results", {}).get("metric_name"),
                }
            )
        context_blob = _context_str(context_summary)
        if context_blob:
            span_attributes["context.summary"] = context_blob

        span_cm = (
            tracer.span("nomad.bench.iteration", span_attributes)
            if tracer
            else nullcontext()
        )

        with span_cm as span:
            if tracer and span and history:
                tracer.set_attributes(
                    span,
                    {
                        "prompt.history": json.dumps(_history_window(history, history_window)),
                    },
                )
            last_results = env.run_train()
            if tracer and span:
                tracer.set_attributes(
                    span,
                    {
                        "benchmark.metric": last_results.get("metric_value"),
                        "benchmark.metric_name": last_results.get("metric_name"),
                        "benchmark.mae": last_results.get("metrics", {}).get("mae"),
                        "benchmark.rmse": last_results.get("metrics", {}).get("rmse"),
                        "benchmark.r2": last_results.get("metrics", {}).get("r2"),
                        "benchmark.history_length": len(history),
                    },
                )

        _record_history_entry(
            history,
            step=step,
            config=config,
            results=last_results,
            proposal_source=proposal_source,
            context_summary=context_summary,
        )
        print("Result:", json.dumps(last_results, indent=2))

    final = {
        "final_metric": last_results.get("metric_value"),
        "final_metric_name": last_results.get("metric_name"),
        "final_config": config,
        "num_steps": num_steps,
        "history": history,
        "context_summary": context_summary or {},
    }
    print("\n===> Final NOMAD summary")
    print(json.dumps(final, indent=2))
    return final


if __name__ == "__main__":
    print(json.dumps(run_nomad_bench(), indent=2))

