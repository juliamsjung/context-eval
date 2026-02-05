# Toy Workspace Directory

This directory contains dataset preparation artifacts and runtime files for the Toy benchmark.

**Note:** Files in this directory are dataset preparation or debug artifacts and are not consumed by ContextEval at runtime, except for:
- `config.json` - Baseline ML hyperparameter configuration (copied to `run_config.json` at start)
- `train.py` - Training script executed during benchmark runs

Other files (`all_results.json`, etc.) are debug/metadata files created during execution and are not read by the benchmark execution code.
