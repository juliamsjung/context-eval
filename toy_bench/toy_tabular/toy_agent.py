from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

from code import get_env_var  # Reuse existing .env handling
from toy_bench.toy_tabular.toy_env import ToyTabularEnv

from logging_utils import start_run
import hashlib

TOY_TABULAR = Path(__file__).resolve().parent
ALL_RESULTS_PATH = TOY_TABULAR / "workspace" / "all_results.json"

logger = logging.getLogger(__name__)


def _safe_parse_json(text: str) -> Dict[str, Any]:
    """Attempt to parse JSON, falling back to extracting the first object."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _propose_config(
    config: Dict[str, Any],
    results: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Use the OpenAI chat API to propose new hyperparameters."""
    if not OPENAI_AVAILABLE:
        return None

    api_key = get_env_var("OPENAI_API_KEY")
    if not api_key:
        return None

    history_lines = "\n".join(
        f"- step {idx}: accuracy={entry['accuracy']:.4f}, "
        f"C={entry['config']['C']}, max_iter={entry['config']['max_iter']}"
        for idx, entry in enumerate(history, start=1)
    )
    if not history_lines:
        history_lines = "- baseline only"

    user_prompt = (
        "You are adjusting hyperparameters for logistic regression on a fixed synthetic dataset.\n"
        f"Current config:\n{json.dumps({'C': config['C'], 'max_iter': config['max_iter']}, indent=2)}\n\n"
        f"Latest results:\n{json.dumps(results, indent=2)}\n\n"
        f"History:\n{history_lines}\n\n"
        "Return JSON with numeric keys 'C' and 'max_iter'. Keep values positive and reasonable."
    )

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=200,
        messages=[
            {
                "role": "system",
                "content": "You propose new logistic regression hyperparameters based on past evaluations.",
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

    filtered: Dict[str, Any] = {}
    if "C" in proposal:
        filtered["C"] = max(0.01, float(proposal["C"]))
    if "max_iter" in proposal:
        filtered["max_iter"] = max(10, int(proposal["max_iter"]))
    return filtered or None


def _fallback_config(current: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
    """Deterministic heuristic if LLM proposals are unavailable."""
    factor = 1.4 if step_idx % 2 == 0 else 0.8
    new_C = max(0.01, min(10.0, current["C"] * factor))
    new_iter = max(50, min(1000, current["max_iter"] + 50))
    return {"C": round(new_C, 4), "max_iter": new_iter}


def run_toy_tabular(
    num_steps: int = 3,
    tracer: Any = None,
    context_policy: Any = None,  # Placeholder for future policy hooks
) -> Dict[str, Any]:
    """Run the toy tabular tuning loop for a fixed number of steps."""
    env = ToyTabularEnv()
    config = env.read_config()
    history: List[Dict[str, Any]] = []

    # --- RunLogger setup ---
    config_json = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha1(config_json.encode("utf-8")).hexdigest()[:10]
    agent_id = "toy_logreg"
    dataset_id = "toy_tabular"
    task_id = "toy_tabular"
    # For simplicity, seed can be fixed or fetched from config if available
    seed = config.get("seed", 0)

    logger = start_run(
        task_id=task_id,
        dataset_id=dataset_id,
        agent_id=agent_id,
    )
    logger.log_run_start(
        config_hash=config_hash,
        max_steps=num_steps,
        seed=seed,
        notes="Toy tabular run with RunLogger"
    )

    print("===> Baseline run")
    baseline_results = env.run_train()
    history.append({"config": config.copy(), "accuracy": baseline_results["accuracy"]})
    last_results = baseline_results

    with open(ALL_RESULTS_PATH, 'w') as f:
        json.dump([], f, indent=2)

    logger.log_op(
        "op.train",
        step_idx=0,
        details={
            "C": config["C"],
            "max_iter": config["max_iter"],
            "accuracy": baseline_results["accuracy"],
            "source": "baseline"
        },
    )

    for step in range(1, num_steps + 1):
        print(f"\n===> Step {step}/{num_steps}")
        proposal = _propose_config(config, last_results, history)
        if not proposal:
            proposal = _fallback_config(config, step)
            print("Using heuristic proposal:", proposal)
        else:
            print("LLM proposal:", proposal)
        
        logger.log_op(
            "op.config_proposal",
            step_idx=step,
            details={
                "proposal": proposal,
                "using_llm": proposal is not None and step > 0,
            },
        )

        config.update({k: proposal[k] for k in ("C", "max_iter") if k in proposal})
        env.write_config(config)

        span_cm = (
            tracer.span(
                "toybench.toy_tabular.iteration",
                {
                    "step": step,
                    "num_steps": num_steps,
                    "C": config["C"],
                    "max_iter": config["max_iter"],
                },
            )
            if tracer
            else nullcontext()
        )

        with span_cm as span:
            last_results = env.run_train()
            if tracer and span:
                tracer.set_attributes(
                    span,
                    {
                        "benchmark.accuracy": last_results["accuracy"],
                        "benchmark.C": config["C"],
                        "benchmark.max_iter": config["max_iter"],
                    },
                )
            logger.log_op(
                "op.train",
                step_idx=step,
                details={
                    "C": config["C"],
                    "max_iter": config["max_iter"],
                    "accuracy": last_results["accuracy"],
                },
            )
            logger.log_step_summary(
                step_idx=step,
                details={
                    "metrics": {"accuracy": last_results["accuracy"]},
                    "config": {"C": config["C"], "max_iter": config["max_iter"]},
                    "decision": {
                        "source": "llm" if proposal is not None and step > 0 else "heuristic"
                    }
                }
            )

        history.append({"config": config.copy(), "accuracy": last_results["accuracy"]})
        print("Result:", json.dumps(last_results, indent=2))

    final = {
        "final_accuracy": last_results["accuracy"],
        "final_config": config,
        "num_steps": num_steps,
    }
    print("\n===> Final summary")
    print(json.dumps(final, indent=2))
    logger.log_run_end(
        status="success",
        final_metric=last_results["accuracy"],
        best_step_idx=step,
        n_steps=num_steps,
    )
    return final


if __name__ == "__main__":
    print(json.dumps(run_toy_tabular(), indent=2))


