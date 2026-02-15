# context-eval

## Overview

ContextEval is a benchmarking framework that isolates and studies the causal effects of context visibility on LLM agent behavior in iterative machine learning workflows, without modifying model architecture or prompt structure.

![Architecture](figs/new_architecture.png)

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

Results saved to `traces/{benchmark}/{timestamp}/`.

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
   - [Leaf Classification](https://www.kaggle.com/c/leaf-classification) - Click "Late Submission"
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

### Leaf Dataset (Species Classification - Image Data (Tabular Data of the Image's Features))

```bash
# 1. Fetch from Kaggle
python3 scripts/fetch_leaf.py

# 2. Unzip the nested archives
cd kaggle-data/leaf/raw
unzip -o leaf-classification.zip
unzip -o train.csv.zip
unzip -o test.csv.zip
cd ../../..

# 3. Prepare offline artifacts
python3 scripts/prepare_leaf.py --float32
```

Prepared data saved to `src/benchmarks/leaf/workspace/`.

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
| `--show-resources` | off | Include resource usage (tokens, cost, latency) in LLM prompt |
| `--history-window` | 0 | Number of history entries to include (0=none, 5=recommended) |

### Experiment Settings

| Flag | Default | Description |
|------|---------|-------------|
| `--num-steps` | 3 | Number of optimization iterations |
| `--seed` | 0 | Random seed for reproducibility |
| `--run-id` | auto | Custom run identifier |
| `--model` | gpt-4o-mini | LLM model to use |
| `--temperature` | 0 | LLM temperature setting |

### Developer Tools

| Flag | Default | Description |
|------|---------|-------------|
| `--verbose` | off | Enable step-by-step logging |
| `--debug-show-llm` | off | Print full LLM request and response for debugging |

## Running Benchmarks

### Available Benchmarks

| Script | Dataset | Task Type |
|--------|---------|-----------|
| `run_toy_bench.py` | Synthetic | Logistic regression tuning |
| `run_nomad_bench.py` | NOMAD 2018 | Materials science regression |
| `run_jigsaw_bench.py` | Jigsaw Toxic Comments | Multi-label text classification |

---

## Behavioral Stability Metrics

Every benchmark run automatically computes **stability metrics** over the agent's full configuration trajectory. These are behavioral measures of _how_ the agent searched, independent of the final benchmark score.

Results appear under `stability_metrics` in the run output dict and in trace logs.

### Configuration Churn

Measures how much the agent changes its configuration at each step. Each hyperparameter contributes independently:

- **Numerical** (`int` / `float`): fractional difference relative to the larger magnitude — `|a − b| / (max(|a|, |b|) + ε)` — giving a value in `[0, 1]` per parameter.
- **Categorical / string**: binary penalty of `1.0` if the value changed, `0.0` if not.
- **Missing in one config**: treated as a categorical change (`1.0`).

Distances are summed across all keys to produce a per-step churn score, then averaged over the run.

### Instability Score (Oscillation)

Detects when the agent revisits a previously-seen configuration. Each step whose hyperparameter dict (compared by canonical JSON hash) has been seen before in the trace is counted as a _repeated_ step.

```
instability_score = repeated_steps / total_steps
```

A score of `0.0` means every step was a novel configuration; `1.0` means every step revisited a prior configuration.

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `instability_score` | float | Fraction of steps that revisit a prior configuration |
| `average_churn` | float | Mean step-to-step config distance across the run |
| `total_churn` | float | Unnormalised sum of all pairwise distances |
| `total_steps` | int | Steps in the trace (including baseline step 0) |
| `churn_steps` | int | Consecutive pairs evaluated (`total_steps − 1`) |

### Implementation

```
src/metrics/
├── __init__.py
└── stability.py   # StabilityMetric class
```

`StabilityMetric` is stateless and can be used standalone:

```python
from src.metrics import StabilityMetric

metric = StabilityMetric()

# Pairwise distance between two configs
dist, changed_keys = metric.calculate_config_distance(config_a, config_b)

# Full trace evaluation (list of dicts with a 'config' key)
result = metric.evaluate_trace(trace_history)
# {'instability_score': 0.25, 'average_churn': 0.6, 'total_churn': 1.8,
#  'total_steps': 4, 'churn_steps': 3}
```

