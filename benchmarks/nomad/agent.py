from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

from code import get_env_var
from benchmarks.nomad.env import NomadEnv
from agent_system import AgentRunner, AgentRunLogger, build_nomad_tools, create_policy

from logging_utils import start_run
import hashlib

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
    clarifications: Optional[List[Dict[str, Any]]] = None,
) -> None:
    history.append(
        {
            "step": step,
            "config": config.copy(),
            "results": results.copy(),
            "proposal_source": proposal_source,
            "context": context_summary or {},
            "clarifications": clarifications or [],
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
                "clarifications": entry.get("clarifications", []),
            }
        )
    return payload


def _clarification_hints_from_history(history: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    hints: Dict[str, str] = {}
    for entry in history:
        for clar in entry.get("clarifications", []) or []:
            q = str(clar.get("question") or "").strip().lower()
            ans = clar.get("answer") or clar.get("observation") or ""
            if q and ans:
                hints[q] = ans
    return hints


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


def _parse_proposal_from_answer(answer: str) -> Dict[str, Any]:
    """Attempt to parse a proposal dict from the agent's final answer."""
    if not answer:
        return {}
    text = answer.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                text = part
                break
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        if "proposal" in parsed and isinstance(parsed["proposal"], dict):
            return _sanitize_proposal(parsed["proposal"])
        return _sanitize_proposal(parsed)
    return {}


def _maybe_build_agent_runner(
    *,
    reasoning_mode: str,
    policy_type: str,
    config: Optional[Dict[str, Any]],
    context_summary: Dict[str, Any],
    tracer: Any,
) -> tuple[Optional[Any], Optional[AgentRunner]]:
    if reasoning_mode != "agentic":
        return None, None

    cfg = config or {}
    context_policies = cfg.get("context_policies", {})
    policy_overrides = context_policies.get(policy_type, {})
    policy = create_policy(policy_type, policy_overrides)
    agent_cfg = cfg.get("agentic", {})
    agent_cfg.setdefault("dataset_id", "nomad")
    agent_cfg.setdefault("agent_id", cfg.get("project_name", "nomad_agent"))
    # Disable legacy agent_runs.jsonl; traces are written via RunLogger + agent iteration events.
    logger = None
    tools = build_nomad_tools(
        context_summary=context_summary,
        retrieval_config=policy.config,
        clarifier_defaults=agent_cfg.get("clarifier_defaults"),
    )
    runner = AgentRunner(
        policy=policy,
        tools=tools,
        logger=logger,
        tracer=tracer,
        config=agent_cfg,
    )
    return policy, runner


def _propose_config(
    config: Dict[str, Any],
    results: Dict[str, Any],
    history: List[Dict[str, Any]],
    *,
    context_summary: Optional[Dict[str, Any]] = None,
    history_window: int = DEFAULT_HISTORY_WINDOW,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not OPENAI_AVAILABLE:
        return None, None

    api_key = get_env_var("OPENAI_API_KEY")
    if not api_key:
        return None, None

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
    t0 = datetime.utcnow().timestamp()
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
    latency = datetime.utcnow().timestamp() - t0
    content = resp.choices[0].message.content or ""
    usage: Dict[str, Any] = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens,
        "latency_sec": latency,
    }
    try:
        proposal = _safe_parse_json(content)
    except Exception:
        logger.warning("LLM response was not valid JSON: %s", content)
        return None, usage

    sanitized = _sanitize_proposal(proposal)
    return sanitized or None, usage


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
    policy_type: str = "short_context",
    reasoning_mode: str = "controller",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    env = NomadEnv()
    model_config = env.read_config()
    context_summary = env.read_context()
    history: List[Dict[str, Any]] = []

    config_json = json.dumps(model_config, sort_keys=True)
    config_hash = hashlib.sha1(config_json.encode("utf-8")).hexdigest()[:10]
    agent_id = model_config.get("model_id", "nomad_agent")
    dataset_id = "nomad"
    task_id = "nomad"
    seed = model_config.get("random_seed", 0)

    logger = start_run(
        task_id=task_id,
        dataset_id=dataset_id,
        agent_id=agent_id,
    )
    logger.log_run_start(
        config_hash=config_hash,
        max_steps=num_steps,
        seed=seed,
        notes="NOMAD run with RunLogger"
    )

    run_base_id = logger.run_id

    print("===> Baseline run (NOMAD)")
    baseline_results = env.run_train()
    _record_history_entry(
        history,
        step=0,
        config=model_config,
        results=baseline_results,
        proposal_source="baseline",
        context_summary=context_summary,
    )
    last_results = baseline_results

    logger.log_op(
        "op.train",
        step_idx=0,
        details={
            "config": model_config,
            "results": baseline_results,
            "source": "baseline",
        },
    )

    policy_obj, agent_runner = _maybe_build_agent_runner(
        reasoning_mode=reasoning_mode,
        policy_type=policy_type,
        config=config,
        context_summary=context_summary,
        tracer=tracer,
    )

    for step in range(1, num_steps + 1):
        print(f"\n===> NOMAD Step {step}/{num_steps}")
        agent_metrics: Optional[Dict[str, Any]] = None
        clarifications: List[Dict[str, Any]] = []

        if reasoning_mode == "agentic" and agent_runner:
            # Build clarification hints from prior iterations (questions/answers), plus defaults.
            carry_hints = _clarification_hints_from_history(history)
            carry_hints.update(
                {
                    "target metric": last_results.get("metric_name", "mae").lower(),
                    "dataset": "nomad",
                }
            )
            agent_state = {
                "context_excerpt": json.dumps(context_summary, sort_keys=True)[:2000],
                "history": _history_window(history, history_window),
                "clarification_hints": carry_hints,
            }
            task_input = (
                f"Iteration {step}/{num_steps}: improve HistGradientBoostingRegressor "
                f"for the NOMAD benchmark. Current metric {last_results.get('metric_value')} "
                f"({last_results.get('metric_name')}). Propose new hyperparameters."
            )
            agent_result = agent_runner.run(
                task_input=task_input,
                state=agent_state,
                run_id=run_base_id,
                task_id="nomad",
                reasoning_mode=reasoning_mode,
                iteration_idx=step,
            )
            agent_metrics = agent_result.metrics
            proposal = _sanitize_proposal(agent_result.structured_output.get("proposal", {}))
            clarifications = agent_result.iteration_entry.get("clarifying_questions", [])
            if not proposal:
                proposal = _parse_proposal_from_answer(agent_result.final_answer)
            if proposal:
                proposal_source = f"agent:{policy_obj.name if policy_obj else policy_type}"
                print("Agent proposal:", proposal)
            else:
                proposal, llm_usage = _propose_config(
                    model_config,
                    last_results,
                    history,
                    context_summary=context_summary,
                    history_window=history_window,
                )
                proposal_source = "agent_fallback_llm" if proposal else "agent_fallback_heuristic"
                if not proposal:
                    proposal = _fallback_config(model_config, step)
                agent_metrics = llm_usage
        else:
            proposal, llm_usage = _propose_config(
                model_config,
                last_results,
                history,
                context_summary=context_summary,
                history_window=history_window,
            )
            proposal_source = "llm"
            if not proposal:
                proposal = _fallback_config(model_config, step)
                proposal_source = "heuristic"
            agent_metrics = llm_usage

        if proposal_source.startswith("agent") and not proposal:
            proposal = _fallback_config(model_config, step)
            proposal_source = "agent_fallback_heuristic"

        print(f"{proposal_source.upper()} proposal:", proposal)

        logger.log_op(
            "op.config_proposal",
            step_idx=step,
            details={
                "proposal": proposal,
                "using_llm": proposal is not None and proposal_source.startswith("llm"),
                "proposal_source": proposal_source,
                "agent_metrics": agent_metrics or {},
            },
        )

        model_config.update(proposal)
        env.write_config(model_config)

        span_attributes = {
            "step": step,
            "num_steps": num_steps,
            "proposal_source": proposal_source,
            **{f"config.{k}": model_config.get(k) for k in PARAM_BOUNDS.keys()},
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
        logger.log_op(
            "op.train",
            step_idx=step,
            details={
                "config": model_config,
                "results": last_results,
                "source": proposal_source,
            },
        )

        logger.log_step_summary(

            step_idx=step,
            details={
                "metrics": last_results.get("metrics", {}),
                "config": model_config,
                "decision": {"source": proposal_source},
                "context": context_summary,
            },
        )

        _record_history_entry(
            history,
            step=step,
            config=model_config,
            results=last_results,
            proposal_source=proposal_source,
            context_summary=context_summary,
            clarifications=clarifications,
        )
        print("Result:", json.dumps(last_results, indent=2))

    final = {
        "final_metric": last_results.get("metric_value"),
        "final_metric_name": last_results.get("metric_name"),
        "final_config": model_config,
        "num_steps": num_steps,
        "history": history,
        "context_summary": context_summary or {},
    }
    print("\n===> Final NOMAD summary")
    print(json.dumps(final, indent=2))

    logger.log_run_end(
        status="success",
        final_metric=last_results.get("metric_value"),
        best_step_idx=step,
        n_steps=num_steps,
    )
    return final


if __name__ == "__main__":
    print(json.dumps(run_nomad_bench(), indent=2))

