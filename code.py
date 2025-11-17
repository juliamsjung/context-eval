"""
Toy LLM system with context-only controller and JSONL tracing.

Phase 1 features:
- Deterministic, context-only controller (scaffold/memory/tool-policy)
- JSONL trace logging for every step
- Minimal success check: valid JSON output with required keys
- Stub mode when OPENAI_API_KEY is not set (deterministic fallback)

Requires:
  pip install openai (optional, for real API calls)
  Add OPENAI_API_KEY="..." to .env (optional)
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import os
import time
import copy
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env_file(env_path: Optional[str] = None) -> Dict[str, str]:
    """Load environment variables from a .env file if present."""
    path = Path(env_path) if env_path else DEFAULT_ENV_PATH
    if not path.exists():
        return {}
    
    loaded_vars: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            
            if "=" not in line:
                continue
            
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            
            if not key:
                continue
            
            if value and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            
            loaded_vars[key] = value
    
    return loaded_vars


# Load environment variables at import time so OPENAI_API_KEY is available
_ENV_VARS = load_env_file()


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch a value from the loaded .env data."""
    return _ENV_VARS.get(key, default)


@dataclass
class Result:
    success: bool
    output: str
    metrics: Dict[str, Any]
    trace_path: Optional[str] = None


@dataclass
class ContextState:
    """Context state for the controller (scaffold, memory, tool-policy)."""
    scaffold: str  # System-level instructions/scaffolding
    memory: List[Dict[str, Any]]  # Conversation/memory history
    tool_policy: Dict[str, Any]  # Tool usage policy/rules
    iteration: int = 0


class TraceLogger:
    """JSONL trace logger for capturing all steps."""
    
    def __init__(self, trace_path: str):
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict[str, Any]] = []
    
    def log_step(
        self,
        step_type: str,
        context: ContextState,
        messages: Optional[List[Dict[str, Any]]] = None,
        usage: Optional[Dict[str, Any]] = None,
        budgets: Optional[Dict[str, Any]] = None,
        stop_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log a step to the trace."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "step_type": step_type,
            "iteration": context.iteration,
            "context": {
                "scaffold": context.scaffold,
                "memory": copy.deepcopy(context.memory),  # Deep copy to avoid reference issues
                "tool_policy": copy.deepcopy(context.tool_policy),  # Deep copy to avoid reference issues
            },
        }
        if messages is not None:
            entry["messages"] = messages
        if usage is not None:
            entry["usage"] = usage
        if budgets is not None:
            entry["budgets"] = budgets
        if stop_reason is not None:
            entry["stop_reason"] = stop_reason
        if metadata is not None:
            entry["metadata"] = metadata
        
        self.entries.append(entry)
    
    def flush(self):
        """Write all entries to JSONL file."""
        with open(self.trace_path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + "\n")
    
    def get_entries(self) -> List[Dict[str, Any]]:
        """Get all logged entries."""
        return self.entries


class ContextController:
    """Deterministic context-only controller (no fine-tuning)."""
    
    def __init__(self, config: Dict[str, Any], trace_logger: TraceLogger):
        self.config = config
        self.trace_logger = trace_logger
        self.context = ContextState(
            scaffold=config.get("context", {}).get("scaffold", "You are a precise, concise assistant."),
            memory=config.get("context", {}).get("memory", []),
            tool_policy=config.get("context", {}).get("tool_policy", {"enabled": False}),
            iteration=0,
        )
    
    def update_memory(self, entry: Dict[str, Any]):
        """Update memory with a new entry."""
        self.context.memory.append(entry)
        self.trace_logger.log_step(
            "memory_update",
            self.context,
            metadata={"memory_entry": entry},
        )
    
    def increment_iteration(self):
        """Increment iteration counter."""
        self.context.iteration += 1
    
    def get_context(self) -> ContextState:
        """Get current context state."""
        return self.context


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    return json.loads(Path(path).read_text())


def _build_messages(context: ContextState, user_input: str, iteration_history: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Build message list from context and user input, including iteration history."""
    messages = []
    
    # Add system message from scaffold
    if context.scaffold:
        messages.append({"role": "system", "content": context.scaffold})
    
    # Add memory/history
    for mem_entry in context.memory:
        if "role" in mem_entry and "content" in mem_entry:
            messages.append(mem_entry)
    
    # Add iteration history if provided (for iterative tasks like Semantle)
    if iteration_history:
        history_text = "\n\n## Previous Attempts:\n"
        for i, hist in enumerate(iteration_history, 1):
            guess = hist.get("guess", "N/A")
            similarity = hist.get("similarity", "N/A")
            feedback = hist.get("feedback", "")
            history_text += f"Attempt {i}: Guess '{guess}' → Similarity: {similarity} ({feedback})\n"
        messages.append({"role": "user", "content": history_text})
    
    # Add current user input
    messages.append({"role": "user", "content": user_input})
    
    return messages


def _check_json_success(output: str, required_keys: List[str]) -> tuple[bool, Optional[str]]:
    """Check if output is valid JSON with required keys."""
    try:
        parsed = json.loads(output)
        if not isinstance(parsed, dict):
            return False, "Output is not a JSON object"
        
        missing_keys = [key for key in required_keys if key not in parsed]
        if missing_keys:
            return False, f"Missing required keys: {missing_keys}"
        
        return True, None
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"


def _evaluate_task(task_config: Dict[str, Any], guess: str, iteration_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate a task guess and return results (e.g., similarity score for Semantle).
    
    For Semantle-like tasks:
    - Extract the guess from the output
    - Calculate similarity score (stub: deterministic based on guess)
    - Return score and feedback
    """
    task_type = task_config.get("type", "semantle")
    
    if task_type == "semantle":
        # Extract guess from output (could be JSON or plain text)
        guess_word = guess.strip().lower()
        
        # Try to parse as JSON first
        try:
            parsed = json.loads(guess)
            if "guess" in parsed:
                guess_word = str(parsed["guess"]).strip().lower()
        except:
            pass
        
        # Stub similarity calculation (deterministic)
        # In real implementation, this would use word2vec or similar
        target_word = task_config.get("target_word", "example").lower()
        
        # Simple stub similarity: higher score for closer words
        if guess_word == target_word:
            similarity = 100.0
            feedback = "Correct! You found the word."
        else:
            # Deterministic stub: calculate based on character overlap
            common_chars = set(guess_word) & set(target_word)
            similarity = min(50.0 + len(common_chars) * 5.0, 99.0)
            
            # Add some deterministic variation based on word length
            length_diff = abs(len(guess_word) - len(target_word))
            similarity -= length_diff * 2.0
            similarity = max(0.0, min(99.0, similarity))
            
            # Provide directional feedback
            if similarity > 70:
                feedback = "Very close! Try words with similar meaning."
            elif similarity > 40:
                feedback = "Getting warmer. Consider the context."
            else:
                feedback = "Not close. Try a different approach."
        
        return {
            "similarity": round(similarity, 2),
            "feedback": feedback,
            "guess": guess_word,
            "target": target_word if similarity >= 100 else None,  # Only reveal if correct
            "correct": similarity >= 100.0,
        }
    else:
        # Default stub evaluation
        return {
            "score": 0.5,
            "feedback": "Task evaluation not implemented",
            "guess": guess,
        }


def _stub_run(config: Dict[str, Any], context: ContextState, trace_logger: TraceLogger, iteration_history: Optional[List[Dict[str, Any]]] = None) -> tuple[str, Dict[str, Any], str]:
    """Deterministic stub run when API key is not available."""
    task = config.get("task", {})
    user_input = task.get("input", "Say hello in one sentence.")
    output_schema = task.get("output_schema", {})
    required_keys = output_schema.get("required_keys", [])
    
    # Build messages (including iteration history)
    messages = _build_messages(context, user_input, iteration_history)
    
    # Log API call attempt
    trace_logger.log_step(
        "api_call_attempt",
        context,
        messages=messages,
        metadata={"mode": "stub", "reason": "OPENAI_API_KEY not set", "iteration": context.iteration},
    )
    
    # Generate deterministic stub output
    if required_keys:
        # Create a stub JSON response with required keys
        stub_output = {key: f"stub_{key}_value" for key in required_keys}
        output_text = json.dumps(stub_output, indent=2)
    else:
        # For iterative tasks, generate a guess
        task_type = task.get("type", "")
        if task_type == "semantle":
            # Generate deterministic guesses that get progressively closer
            target_word = task.get("target_word", "example").lower()
            iteration_num = len(iteration_history)
            
            # Deterministic stub guesses that approach the target
            stub_guesses = [
                "word", "term", "sample", "instance", "case", 
                "illustration", "demonstration", "specimen", "model", "example"
            ]
            
            # Cycle through guesses, eventually landing on target
            if iteration_num < len(stub_guesses):
                guess = stub_guesses[iteration_num]
                # If we're close to target position, use target
                if guess == target_word or iteration_num >= len(stub_guesses) - 1:
                    output_text = target_word
                else:
                    output_text = guess
            else:
                # After cycling, return target
                output_text = target_word
        else:
            # Simple text response
            output_text = f"Stub response to: {user_input}"
    
    # Simulate usage
    usage = {
        "input_tokens": len(str(messages)) // 4,  # Rough estimate
        "output_tokens": len(output_text) // 4,
        "total_tokens": (len(str(messages)) + len(output_text)) // 4,
    }
    
    # Determine stop reason
    stop_reason = "stub_completion"
    
    # Log completion
    trace_logger.log_step(
        "api_response",
        context,
        messages=[{"role": "assistant", "content": output_text}],
        usage=usage,
        stop_reason=stop_reason,
        metadata={"mode": "stub", "iteration": context.iteration},
    )
    
    return output_text, usage, stop_reason


def _real_run(
    config: Dict[str, Any],
    context: ContextState,
    trace_logger: TraceLogger,
    api_key: str,
    iteration_history: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, Dict[str, Any], str]:
    """Real API run using OpenAI SDK."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not available. Install with: pip install openai")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")
    
    model = config.get("model", "gpt-4o-mini")
    temperature = float(config.get("temperature", 0.0))
    max_output_tokens = int(config.get("max_output_tokens", 256))
    task = config.get("task", {})
    user_input = task.get("input", "Say hello in one sentence.")
    
    # Build messages (including iteration history)
    messages = _build_messages(context, user_input, iteration_history)
    
    # Log API call attempt
    trace_logger.log_step(
        "api_call_attempt",
        context,
        messages=messages,
        metadata={"model": model, "temperature": temperature, "max_output_tokens": max_output_tokens, "iteration": context.iteration},
    )
    
    # Make API call
    client = OpenAI(api_key=api_key)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        elapsed = time.time() - t0
        
        # Extract response
        output_text = resp.choices[0].message.content or ""
        stop_reason = resp.choices[0].finish_reason or "unknown"
        
        # Extract usage
        usage = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
        
        # Log completion
        trace_logger.log_step(
            "api_response",
            context,
            messages=[{"role": "assistant", "content": output_text}],
            usage=usage,
            stop_reason=stop_reason,
            metadata={"elapsed_sec": round(elapsed, 4), "iteration": context.iteration},
        )
        
        return output_text, usage, stop_reason
        
    except Exception as e:
        # Log error
        trace_logger.log_step(
            "api_error",
            context,
            metadata={"error": str(e), "error_type": type(e).__name__, "iteration": context.iteration},
        )
        raise


def run(config: Dict[str, Any]) -> Result:
    """Run the toy LLM system with context controller and iterative tracing."""
    # Initialize trace logger
    trace_path = config.get("trace_path", "traces/run.jsonl")
    trace_logger = TraceLogger(trace_path)
    
    # Initialize context controller
    controller = ContextController(config, trace_logger)
    context = controller.get_context()
    
    # Get budgets
    budgets = {
        "max_tokens": config.get("max_output_tokens", 256),
        "max_iterations": int(config.get("max_iterations", 1)),
        "max_time_sec": config.get("max_time_sec", None),
    }
    
    # Log initialization
    trace_logger.log_step(
        "init",
        context,
        budgets=budgets,
        metadata={"config": {k: v for k, v in config.items() if k != "trace_path"}},
    )
    
    # Determine provider based on .env configuration
    openai_api_key = get_env_var("OPENAI_API_KEY")
    use_stub = not openai_api_key or config.get("use_stub", False)
    provider = "stub" if use_stub else "openai"
    model = "stub" if use_stub else config.get("model", "gpt-4o-mini")
    
    # Get task configuration
    task = config.get("task", {})
    task_type = task.get("type", "simple")
    is_iterative = task_type in ["semantle"] or budgets["max_iterations"] > 1
    
    # Iteration tracking
    iteration_history: List[Dict[str, Any]] = []
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    success = False
    stop_reason = "max_iterations"
    final_output = ""
    start_time = time.time()
    
    try:
        # Iterative execution loop
        for iteration_num in range(budgets["max_iterations"]):
            controller.increment_iteration()
            context = controller.get_context()
            
            # Check time budget
            if budgets["max_time_sec"]:
                elapsed = time.time() - start_time
                if elapsed > budgets["max_time_sec"]:
                    stop_reason = "time_limit"
                    break
            
            # Build user input (for iterative tasks, include feedback from previous iterations)
            if is_iterative and iteration_history:
                # For iterative tasks, prompt for next guess using previous results
                user_input = task.get("iterative_prompt", "Based on the previous attempts, make your next guess.")
            else:
                user_input = task.get("input", "Say hello in one sentence.")
            
            # Run API call (stub or real) with iteration history
            if use_stub:
                output_text, usage, api_stop_reason = _stub_run(config, context, trace_logger, iteration_history)
            else:
                output_text, usage, api_stop_reason = _real_run(
                    config,
                    context,
                    trace_logger,
                    openai_api_key,
                    iteration_history,
                )
            
            # Accumulate usage
            total_usage["input_tokens"] += usage.get("input_tokens", 0)
            total_usage["output_tokens"] += usage.get("output_tokens", 0)
            total_usage["total_tokens"] += usage.get("total_tokens", 0)
            
            # Update memory
            controller.update_memory({"role": "user", "content": user_input})
            controller.update_memory({"role": "assistant", "content": output_text})
            
            # Evaluate task result (for iterative tasks like Semantle)
            if is_iterative:
                evaluation = _evaluate_task(task, output_text, iteration_history)
                iteration_history.append(evaluation)
                
                # Log evaluation result
                trace_logger.log_step(
                    "task_evaluation",
                    context,
                    metadata={
                        "iteration": iteration_num + 1,
                        "evaluation": evaluation,
                    },
                )
                
                # Check success condition
                if evaluation.get("correct", False):
                    success = True
                    stop_reason = "success"
                    final_output = output_text
                    break
                elif evaluation.get("similarity", 0) >= task.get("success_threshold", 100.0):
                    success = True
                    stop_reason = "threshold_reached"
                    final_output = output_text
                    break
                
                # Prepare feedback for next iteration
                feedback_msg = f"Your guess '{evaluation.get('guess', '')}' scored {evaluation.get('similarity', 0):.2f}. {evaluation.get('feedback', '')}"
                controller.update_memory({"role": "user", "content": feedback_msg})
                
                final_output = output_text
            else:
                # Non-iterative task: check success (valid JSON with required keys)
                output_schema = task.get("output_schema", {})
                required_keys = output_schema.get("required_keys", [])
                
                if required_keys:
                    success, error_msg = _check_json_success(output_text, required_keys)
                    if not success:
                        trace_logger.log_step(
                            "validation_failure",
                            context,
                            metadata={"error": error_msg, "output": output_text[:200]},
                        )
                        stop_reason = "validation_failed"
                    else:
                        success = True
                        stop_reason = "validation_passed"
                else:
                    # Fallback: check if output contains expected strings
                    success_contains = task.get("success_contains", [])
                    if success_contains:
                        lower = output_text.lower()
                        success = all(s.lower() in lower for s in success_contains)
                        stop_reason = "contains_check"
                    else:
                        success = True  # Default to success if no checks specified
                        stop_reason = "no_checks"
                
                final_output = output_text
                break  # Non-iterative tasks run once
        
        # Build metrics
        metrics = {
            "provider": provider,
            "model": model,
            "temperature": float(config.get("temperature", 0.0)),
            "max_output_tokens": budgets["max_tokens"],
            "input_tokens": total_usage["input_tokens"],
            "output_tokens": total_usage["output_tokens"],
            "total_tokens": total_usage["total_tokens"],
            "stop_reason": stop_reason,
            "iterations_completed": context.iteration,
            "success": success,
            "elapsed_sec": round(time.time() - start_time, 4),
        }
        
        # Add iteration history for iterative tasks
        if is_iterative:
            metrics["iteration_history"] = iteration_history
        
        # Log final result
        trace_logger.log_step(
            "final_result",
            context,
            usage=total_usage,
            budgets=budgets,
            stop_reason=stop_reason,
            metadata={"success": success, "metrics": metrics, "iteration_history": iteration_history if is_iterative else None},
        )
        
        # Flush trace
        trace_logger.flush()
        
        return Result(
            success=success,
            output=final_output,
            metrics=metrics,
            trace_path=trace_path,
        )
        
    except Exception as e:
        # Log error
        trace_logger.log_step(
            "error",
            context,
            metadata={"error": str(e), "error_type": type(e).__name__, "iteration": context.iteration},
        )
        trace_logger.flush()
        raise
