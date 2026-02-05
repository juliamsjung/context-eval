# NOMAD Workspace Directory

This directory contains dataset preparation artifacts and runtime files for the NOMAD benchmark.

**Note:** Files in this directory are dataset preparation or debug artifacts and are not consumed by ContextEval at runtime, except for:
- `config.json` - Baseline ML hyperparameter configuration (copied to `run_config.json` at start)
- `train.py` - Training script executed during benchmark runs
- `task_description.txt` - Loaded if `--show-task` flag is used
- `metric_description.txt` - Loaded if `--show-metric` flag is used

Other files (`dataset_context.json`, `prepared_meta.json`, etc.) are metadata created during dataset preparation and are not read by the benchmark execution code.
