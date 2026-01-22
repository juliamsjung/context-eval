# DSC180A-Q1Project

## Overview
Our project aims to explore how to evaluate LLM systems with a focus on how context policies shapes their performance. Our main research question is
>How do context policies shape the reliability and efficiency of LLM agents that design, run, and interpret ML experiments?


## Task: ML Experimentation
Our goal for Q1 is to build a framework to systematically evaluate how different context policies (history management, summarization, retrieval, structured state, budget-aware mixing) affect LLM-based ML experimentation. We evaluate agents **as ML experimenters**. An agent plans, writes code, calls tools, and manages files to complete compact ML tasks. Heavily inspired by Stanford's MLAgentBench (https://arxiv.org/pdf/2310.03302) and OpenAI's MLEBench (https://arxiv.org/pdf/2410.07095).

![](figs/new_architecture.png)

**Task format:**
- **Actions:** edit files, run tests/metrics, parse tracebacks, update prompt/tool/memory.
- **Modalities:** start with **text/tabular**; optional image/vision later.
- **Stop criteria:** produce a **valid structured output** (e.g., metrics JSON) or meet a **task-native target** (e.g., ≥10% over a starter baseline) within token/time/iteration budgets.

## Quick Start

```bash
# Run Toy benchmark (simple logistic regression tuning)
python run_toy_bench.py --config config.json --num-steps 3

# Run NOMAD benchmark (materials science regression)
python run_nomad_bench.py --config config.json --num-steps 3
```

## Setup

### Create & activate a virtual environment
Using a venv keeps dependencies pinned to the same versions across machines. Run the following once per clone:

```bash
python3 -m venv venv
source venv/bin/activate         # macOS/Linux
# .\venv\Scripts\activate        # Windows PowerShell
```

Every time you work in the repo, activate the venv first.

### Environment Variables
Copy the provided `.env` template (or create one) in the repo root. The code loads it automatically on import, so no manual `export` is needed. Keep this file untracked.

#### 1. Install dependencies
```bash
pip install openai kaggle pandas scikit-learn
```

#### 2. Configure `.env` file
Required keys:
```
OPENAI_API_KEY=<OpenAI key>
```

---

## Project Structure

```
DSC180A-Q1Project/
├── README.md                           # This file
├── config.json                         # Main configuration
├── .env                                # API keys (not tracked)
│
├── run_nomad_bench.py                  # CLI entry point for NOMAD
├── run_toy_bench.py                    # CLI entry point for Toy
│
├── src/                                # Main source package
│   ├── utils/                          # Shared utilities
│   │   ├── config.py                   # load_config, get_env_var
│   │   └── logging.py                  # RunLogger, start_run
│   │
│   ├── agent/                          # Agent system
│   │   ├── runner.py                   # AgentRunner (ReAct-style)
│   │   ├── policies.py                 # Context policies (short/long)
│   │   └── tools.py                    # Tool definitions
│   │
│   └── benchmarks/                     # All benchmarks
│       ├── base.py                     # BaseBenchmark abstract class
│       ├── nomad/                      # NOMAD benchmark
│       │   ├── benchmark.py            # NomadBenchmark class
│       │   └── env.py                  # NomadEnv
│       └── toy/                        # Toy benchmark
│           ├── benchmark.py            # ToyTabularBenchmark class
│           └── env.py                  # ToyTabularEnv
│
├── benchmarks/nomad/                   # NOMAD workspace (data + training)
├── toy_bench/toy_tabular/              # Toy workspace (data + training)
│
├── scripts/                            # Data preparation utilities
│   ├── fetch_nomad.py
│   └── prepare_nomad.py
│
└── traces/                             # Output traces (JSONL)
    ├── nomad/
    └── toy_tabular/
```

---

## Toy Bench (toy_tabular)

The `toy_bench/toy_tabular` package is a small logistic-regression tuning loop for fast iteration and tracing.

**Dependencies**
```bash
pip install scikit-learn
```

**How to run**
```bash
# Basic run (controller mode)
python run_toy_bench.py --config config.json --num-steps 3

# With agent support (agentic mode)
python run_toy_bench.py --config config.json --num-steps 3 --reasoning-mode agentic --policy-type short_context
```

**Outputs**
- Local traces: `traces/toy_tabular/<run_id>.jsonl`
- CLI prints final accuracy/config

---

## NOMAD Kaggle Benchmark

**What is this dataset?**
The [Nomad 2018 Predict Transparent Conductors](https://www.kaggle.com/competitions/nomad2018-predict-transparent-conductors) challenge asks you to predict the **bandgap energy (eV)** of candidate transparent conductors. Each row (≈2.4k total) represents a simulated crystal with composition and lattice properties.

### Prerequisites
1. **Join the competition** – sign in to Kaggle, open the [Nomad competition page](https://www.kaggle.com/competitions/nomad2018-predict-transparent-conductors), click *Late Submission* and accept the terms/rules.
2. **Kaggle credentials** – place `~/.kaggle/kaggle.json` (or export `KAGGLE_USERNAME`/`KAGGLE_KEY`).
3. **Dependencies** – `pip install kaggle pandas scikit-learn`

### Workflow

1. **Stage raw CSVs**
   ```bash
   mkdir -p kaggle-data/nomad/raw
   kaggle competitions download -c nomad2018-predict-transparent-conductors -p kaggle-data/nomad/raw
   (cd kaggle-data/nomad/raw && unzip nomad2018-predict-transparent-conductors.zip && unzip train.csv.zip && unzip test.csv.zip && unzip sample_submission.csv.zip)
   ```

2. **Prepare arrays + context metadata**
   ```bash
   python scripts/prepare_nomad.py --float32
   ```

3. **Run the benchmark**
   ```bash
   # Controller mode
   python run_nomad_bench.py --config config.json --num-steps 3

   # Agentic mode with short context
   python run_nomad_bench.py --config config.json --num-steps 3 --reasoning-mode agentic --policy-type short_context

   # Agentic mode with long context
   python run_nomad_bench.py --config config.json --num-steps 3 --reasoning-mode agentic --policy-type long_context
   ```

---

## Context Policies and Reasoning Modes

We support explicit context policies and reasoning modes:
- `policy_type`: `short_context` (tight retrieval, encourages clarifications) vs `long_context` (large retrieval budget, fewer clarifications).
- `reasoning_mode`: `agentic` (ReAct-style loop with tools) vs `controller` (legacy single-shot LLM calls).

### Configuration
See `config.json`:
- Top-level `policy_type` and `reasoning_mode` defaults
- `context_policies` block defines retrieval budgets per policy
- `agentic` block defines agent loop parameters

### Examples

**NOMAD benchmark:**
```bash
# Agentic + short context
python run_nomad_bench.py --config config.json --policy-type short_context --reasoning-mode agentic --num-steps 3

# Controller + long context
python run_nomad_bench.py --config config.json --policy-type long_context --reasoning-mode controller --num-steps 3
```

**Toy benchmark:**
```bash
# Agentic + short context
python run_toy_bench.py --config config.json --policy-type short_context --reasoning-mode agentic --num-steps 3

# Controller + long context
python run_toy_bench.py --config config.json --policy-type long_context --reasoning-mode controller --num-steps 3
```

### Batch Experiments

For running large-scale experiments:

```bash
# Custom output directory and run ID
python run_nomad_bench.py --config config.json --num-steps 3 \
  --output-dir outputs/experiment_001 \
  --run-id nomad_agentic_short_seed42 \
  --reasoning-mode agentic \
  --policy-type short_context \
  --seed 42
```

---

## Trace Output

Each run produces JSONL trace files with the following event schema:
```json
{
  "run_id": "nomad_2025-01-22T12-30-45Z",
  "event_type": "op.train",
  "step_idx": 1,
  "timestamp": "2025-01-22T12:30:50Z",
  "task_id": "nomad",
  "dataset_id": "nomad",
  "agent_id": "nomad_agentic_short_context",
  "details": { ... }
}
```

Event types:
- `run.start`: Marks the beginning of a run
- `op.config_proposal`: Config proposal from LLM/agent
- `op.train`: Training result with metrics
- `step.summary`: Per-iteration summary
- `agent.iteration`: Agent step details (agentic mode only)
- `run.end`: Final results

---

## Visualize With Plots
```bash
python toy_bench/toy_tabular/workspace/plot.py
```

> **Tip:** After installing the Kaggle CLI inside the venv, you can verify it's wired up by running `kaggle --version`.
