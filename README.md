# context-eval

## Overview

ContextEval is a benchmarking framework that isolates and studies the causal effects of context visibility on LLM agent behavior in iterative machine learning workflows.

ContextEval does not modify model architecture or prompt phrasing; it manipulates only structured informational access.

ContextEval is not a prompt-engineering toolkit; it is a controlled experimental framework for studying informational exposure.

## Problem Statement

When LLM agents are deployed as autonomous optimizers in machine learning workflows (e.g., hyperparameter tuning), their behavior depends on the information made visible to them at each step.

However, it remains unclear:

> **Which categories of context are necessary and sufficient for stable and efficient iterative optimization?**

ContextEval treats context visibility as an experimental variable. We isolate the causal effect of informational access while holding the following fixed:
- Base LLM
- Dataset and task
- Optimization horizon (number of steps, fixed across all configurations)
- Offline training environment

We vary four context axes:

| Axis | Description |
|------|-------------|
| `show_task` | Task description (problem framing) |
| `show_metric` | Evaluation metric definition |
| `show_bounds` | Explicit hyperparameter bounds (feasible region) |
| `feedback_depth` | Number of prior optimization steps visible to the agent (1 = current only, 5 = current + 4 history) |

By running a full factorial grid (2×2×2×2 = 16 context policies × 3 init qualities = 48 runs per benchmark), we estimate the marginal and interaction effects of each axis on:
- Optimization outcome (best achieved metric)
- Convergence efficiency
- Behavioral stability (oscillation, variance)
- Constraint violations (clamp events)

Importantly, diagnostic signals and resource usage are always recorded in the trace layer but never shown to the agent, ensuring no trace-only signals are exposed to the agent.

## Two-Phase Diagnostic Architecture

To ensure that performance gains are attributed solely to informational visibility—not initialization bias or search space differences—ContextEval uses a two-phase approach modeled after Sequential Model-Based Optimization (SMBO) standards.

### Phase 1: Landscape Characterization

Before running any LLM agent experiments, we map the performance landscape of each benchmark:

1. **Space-filling sampling** — Generate 200 configurations via **Sobol sequences** (quasi-random), providing superior coverage of the hyperparameter space compared to uniform random sampling.
2. **Batch evaluation** — Evaluate each configuration against the benchmark's training pipeline (no LLM involved).
3. **Score distribution** — This establishes the empirical performance distribution, providing a **Random Search baseline** for free.

Parameters spanning multiple orders of magnitude (e.g., `learning_rate`, `C`) are sampled on a **logarithmic scale** to ensure equal coverage across different scales.

### Phase 2: Performance-Stratified Initialization

From the 200 evaluated samples, we select three representative starting configurations:

| Init Quality | Percentile | What It Tests |
|---|---|---|
| **Low** | P25 (median of bottom quartile) | Can the agent recover from a poor starting point? |
| **Neutral** | P50 (overall median) | Can the agent improve from an average start? |
| **High** | P75 (median of top quartile, excl. top 5%) | Can the agent fine-tune near the optimum? |

These three starting configs replace fixed seed-based initialization, allowing us to isolate the agent's **optimization ability** independently of where it starts.

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Landscape Characterization (one-time, no LLM)       │
│                                                                 │
│  Sobol Sampling  →  Batch Evaluation  →  Score Distribution    │
│  (200 configs)      (run_training)       (P25/P50/P75 picks)  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: LLM Agent Experiments (48 runs per benchmark)        │
│                                                                 │
│  For each init quality (low, neutral, high):                   │
│    For each context policy (16 combinations):                  │
│      Run T=10 optimization steps with LLM agent                │
│      Log configs, metrics, clamp events, token usage           │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture Overview

ContextEval enforces strict separation between execution, context projection, and trace logging to guarantee experimental isolation and prevent context leakage.

- **Execution Layer**: Runs optimization steps and training.
- **Context Layer**: Controls what information is exposed to the LLM.
- **Trace Layer**: Logs full system state for analysis (never exposed to the LLM).

```
context-eval/
├── src/
│   ├── benchmarks/          # Execution layer (training + iteration logic)
│   │   ├── base.py
│   │   ├── nomad/
│   │   ├── jigsaw/
│   │   ├── forest/
│   │   └── toy/
│   ├── context/             # Context projection layer (LLM-visible data only)
│   │   ├── axes.py
│   │   ├── builder.py
│   │   ├── formatter.py
│   │   └── schema.py
│   ├── landscape/           # Landscape characterization (Phase 1)
│   │   ├── sampler.py       #   Sobol quasi-random sampling
│   │   ├── runner.py        #   Batch evaluation
│   │   └── selector.py      #   Stratified init selection (P25/P50/P75)
│   ├── trace/               # Trace layer (full observability, never exposed)
│   │   ├── logger.py
│   │   ├── run_summary.py
│   │   └── schema.py
│   └── utils/
├── scripts/
│   ├── run_landscape.py     # Landscape characterization entry point
│   └── run_grid.sh          # Experiment grid runner
├── logs/
│   ├── landscape/           # Landscape results + init configs
│   ├── traces/
│   └── runs/
└── run_*_bench.py
```

## Design Principles

- **Causal Isolation** — Only context visibility is varied.
- **No Prompt Engineering** — Instruction phrasing is fixed across runs.
- **No Context Leakage** — Trace-only signals are never shown to the LLM.
- **Full Observability** — All execution state is recorded for analysis.

## Quick Start

```bash
# 1. Landscape characterization (one-time per benchmark, no LLM needed)
python scripts/run_landscape.py --benchmark <benchmark> --num-samples 200

# 2. Single benchmark run
python run_<benchmark>_bench.py --num-steps 10 --show-task --show-metric

# 3. Full experiment grid (48 runs: 3 init qualities × 16 context policies)
#    Requires landscape characterization (step 1) to have been run first.
./scripts/run_grid.sh <benchmark> --dry-run    # preview commands
./scripts/run_grid.sh <benchmark>              # run full grid
```

Where `<benchmark>` is one of: `nomad`, `jigsaw`, `forest`, `toy`.

Results saved to `logs/` (landscape in `logs/landscape/`, traces in `logs/traces/`, run summaries in `logs/runs/`).

---

## Setup

### 1. Clone & Create Environment

```bash
git clone <repo-url>
cd context-eval
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
| Package | Purpose |
|---------|---------|
| `openai` | LLM API client |
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | ML models and metrics |
| `scipy` | Sobol sequences for landscape characterization |

**Analysis & Visualization:**
| Package | Purpose |
|---------|---------|
| `matplotlib`, `seaborn` | Plotting |
| `jupyter` | Notebook environment |
| `tqdm` | Progress bars |

**Optional:**
| Package | Purpose |
|---------|---------|
| `kaggle` | Dataset fetching (only needed once) |
| `pytest` | Testing |

### 3. Configure API Keys

Create `.env` in repo root:

```
OPENAI_API_KEY=<your-openai-key>
```

---

## Kaggle Dataset Setup

We use Kaggle datasets as data sources. Benchmarks run **offline** using prepared artifacts—no Kaggle access during experiments.

### Prerequisites

1. **Create Kaggle API credentials**:
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings) → Scroll to "API" → Create Legacy API Key
   - This downloads `kaggle.json`. It should be saved in your `/Downloads` folder.

2. **Install the credentials**:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Join the competitions** (accept rules on each page):
   - [NOMAD 2018](https://www.kaggle.com/competitions/nomad2018-predict-transparent-conductors) - Click "Late Submission"
   - [Jigsaw Toxic Comment](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) - Click "Late Submission"
   - [Forest Cover Type](https://www.kaggle.com/c/forest-cover-type-prediction) - Click "Late Submission"
   - [California Housing](https://www.kaggle.com/competitions/playground-series-s3e1) - Click "Late Submission" (Kaggle's Playground Series)

---

### NOMAD Dataset (Materials Science Regression - Tabular Data)

```bash
# 1. Fetch from Kaggle
python3 scripts/fetch_nomad.py

# 2. Unzip the nested archives
cd kaggle-data/nomad/raw
unzip -o nomad2018-predict-transparent-conductors.zip
unzip -o train.csv.zip
unzip -o test.csv.zip
cd ../../..

# 3. Prepare offline artifacts
python3 scripts/prepare_nomad.py --float32
```

Prepared data saved to `src/benchmarks/nomad/workspace/`.

---

### Jigsaw Dataset (Toxic Comment Classification - Text Data)

```bash
# 1. Fetch from Kaggle
python3 scripts/fetch_jigsaw.py

# 2. Unzip the nested archives
cd kaggle-data/jigsaw/raw
unzip -o jigsaw-toxic-comment-classification-challenge.zip
unzip -o train.csv.zip
cd ../../..

# 3. Prepare offline artifacts
python3 scripts/prepare_jigsaw.py
```

Prepared data saved to `src/benchmarks/jigsaw/workspace/`.

---

### Forest Cover Type Dataset (Multi-class Classification - Tabular Data)

```bash
# 1. Fetch from Kaggle
python3 scripts/fetch_forest.py

# 2. Unzip the nested archives
cd kaggle-data/forest/raw
unzip -o forest-cover-type-prediction.zip
cd ../../..

# 3. Prepare offline artifacts
python3 scripts/prepare_forest.py --float32
```

Prepared data saved to `src/benchmarks/forest/workspace/`.

---

### California Housing Dataset (Regression - Tabular Data)

```bash
# 1. Fetch from Kaggle
python3 scripts/fetch_housing.py

# 2. Unzip the nested archives
cd kaggle-data/housing/raw
unzip -o playground-series-s3e1.zip
cd ../../..

# 3. Prepare offline artifacts
python3 scripts/prepare_housing.py --float32
```

Prepared data saved to `src/benchmarks/housing/workspace/`.

---

## CLI Options

### Context Axes
These flags control what information the LLM agent sees:

| Flag | Default | Description |
|------|---------|-------------|
| `--show-task` | off | Include task description in LLM prompt |
| `--show-metric` | off | Include metric description in LLM prompt |
| `--show-bounds` | off | Include parameter bounds (valid ranges) in LLM prompt |
| `--feedback-depth` | 1 | Feedback depth: number of visible outcome signals (1=current only, 5=current+4 history) |

### Experiment Settings

| Flag | Default | Description |
|------|---------|-------------|
| `--num-steps` | 3 | Number of optimization iterations |
| `--seed` | 0 | Random seed for reproducibility |
| `--run-id` | auto | Custom run identifier |
| `--experiment-id` | default | Experiment ID for grouping runs (must be filesystem-safe: `[a-zA-Z0-9_-]+`) |
| `--model` | gpt-4o-mini | LLM model to use |
| `--temperature` | 0 | LLM temperature setting |

### Developer Tools

| Flag | Default | Description |
|------|---------|-------------|
| `--verbose` | off | Enable step-by-step logging |
| `--debug-show-llm` | off | Print full LLM request and response for debugging |
| `--debug-show-diff` | off | Show config changes at each step with score delta |

## Running Benchmarks

### Available Benchmarks

| Script | Dataset | Task Type |
|--------|---------|-----------|
| `run_toy_bench.py` | Synthetic | Logistic regression tuning |
| `run_nomad_bench.py` | NOMAD 2018 | Materials science regression |
| `run_jigsaw_bench.py` | Jigsaw Toxic Comments | Multi-label text classification |
| `run_forest_bench.py` | Forest Cover Type | Multi-class forest cover classification |
| `run_housing_bench.py` | California Housing | Housing price regression |

