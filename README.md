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

By running a full factorial grid (2×2×2×2 = 16 configurations × 3 seeds = 48 runs), we estimate the marginal and interaction effects of each axis on:
- Optimization outcome (best achieved metric)
- Convergence efficiency
- Behavioral stability (oscillation, variance)
- Constraint violations (clamp events)

Importantly, diagnostic signals and resource usage are always recorded in the trace layer but never shown to the agent, ensuring no trace-only signals are exposed to the agent.

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
│   │   └── toy/
│   ├── context/             # Context projection layer (LLM-visible data only)
│   │   ├── axes.py
│   │   ├── builder.py
│   │   ├── formatter.py
│   │   └── schema.py
│   ├── trace/               # Trace layer (full observability, never exposed)
│   │   ├── logger.py
│   │   ├── run_summary.py
│   │   └── schema.py
│   └── utils/
├── scripts/
├── experiments/
├── logs/
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
# Single benchmark runs
python run_toy_bench.py --num-steps 5
python run_nomad_bench.py --num-steps 5 --show-task --show-metric

# Experiment grids (48 runs across all context axis combinations)
./scripts/run_grid.sh toy --dry-run    # preview commands
./scripts/run_grid.sh toy              # run full grid
./scripts/run_grid.sh nomad
```

Results saved to `logs/` (traces in `logs/traces/`, run summaries in `logs/runs/`).

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

