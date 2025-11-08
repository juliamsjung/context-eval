#!/usr/bin/env python3
"""
CLI for the toy LLM system.

Usage:
  python script.py --config config.json
"""
import argparse
import json
from code import load_config, run


def main():
    parser = argparse.ArgumentParser(description="Toy LLM system (OpenAI SDK)")
    parser.add_argument("--config", default="config.json", help="Path to JSON config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run(cfg)
    print(json.dumps(
        {"success": result.success, "output": result.output, "metrics": result.metrics},
        indent=2
    ))


if __name__ == "__main__":
    main()