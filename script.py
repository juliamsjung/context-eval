#!/usr/bin/env python3
"""
CLI for the toy LLM system with context controller and JSONL tracing.

Usage:
  python script.py --config config.json
"""
import argparse
import json
import sys
from pathlib import Path
from code import load_config, run


def main():
    parser = argparse.ArgumentParser(description="Toy LLM system with context controller and tracing")
    parser.add_argument("--config", default="config.json", help="Path to JSON config")
    parser.add_argument("--trace", help="Path to trace output file (overrides config)")
    parser.add_argument(
        "--policy-type",
        choices=["short_context", "long_context"],
        help="Override the context policy selection.",
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=["controller", "agentic"],
        help="Choose whether to run the new agentic loop or the legacy controller.",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.policy_type:
        cfg["policy_type"] = args.policy_type
    if args.reasoning_mode:
        cfg["reasoning_mode"] = args.reasoning_mode
    
    # Override trace path if provided
    if args.trace:
        cfg["trace_path"] = args.trace
    
    # Run the system
    try:
        result = run(cfg)
        
        # Print result
        output = {
            "success": result.success,
            "output": result.output,
            "metrics": result.metrics,
            "trace_path": result.trace_path,
        }
        print(json.dumps(output, indent=2))
        
        # Exit with appropriate code
        sys.exit(0 if result.success else 1)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
