# context-eval

## Overview

ContextEval is a benchmarking framework that studies how context visibility affects LLM optimizer behavior in hyperparameter optimization.

We vary four context axes (task description, metric definition, parameter bounds, feedback depth) while holding fixed the base LLM, prompt template, decoding configuration, and evaluation environment.

ContextEval is not a prompt-engineering toolkit. It is a controlled experimental framework for isolating the effect of what information an agent can see on how it searches.

## Research Question

> **Which categories of context are necessary for effective LLM-based optimization?**

| Context Axis | Description |
|--------------|-------------|
| `show_task` | Task description |
| `show_metric` | Metric definition |
| `show_bounds` | Parameter bounds |
| `feedback_depth` | History depth (1 or 5) |

## Experimental Design

Two-phase approach to isolate context effects from initialization bias.

### Phase 1: Landscape Characterization (one-time, no LLM)

- 256 Sobol-sampled configurations per benchmark
- Establishes performance distribution and random search baseline
- Selects stratified init configs using normalized regret:

| Init | Stratum | Purpose |
|------|---------|---------|
| High | r ≤ 0.20 | Fine-tuning near optimum |
| Neutral | 0.45 ≤ r ≤ 0.55 | Improving from average |
| Low | r ≥ 0.80 | Recovering from poor start |

### Phase 2: LLM Experiments (144 runs per benchmark)

- 3 init qualities × 16 context policies × 3 seeds
- T=10 optimization steps per run

### Random Search Baseline (9 runs per benchmark)

- 3 init qualities × 3 seeds
- Same init configs as LLM for fair comparison
- Zero API cost

## Project Structure

```
context-eval/
├── src/
│   ├── benchmarks/          # Execution layer (training + iteration logic)
│   │   ├── base.py
│   │   ├── nomad/
│   │   ├── forest/
│   │   ├── housing/
│   │   ├── jigsaw/
│   │   └── toy/
│   ├── context/             # Context projection layer (LLM-visible data only)
│   │   ├── axes.py
│   │   ├── builder.py
│   │   ├── formatter.py
│   │   └── schema.py
│   ├── landscape/           # Landscape characterization (Phase 1)
│   │   ├── sampler.py       #   Sobol quasi-random sampling
│   │   ├── runner.py        #   Batch evaluation
│   │   └── selector.py      #   Stratum-based init selection (normalized regret)
│   ├── optimizers/          # Optimizer strategies (strategy pattern)
│   │   ├── base.py          # BaseOptimizer ABC
│   │   └── random.py        # Random search baseline (LLM uses direct path)
│   ├── trace/               # Trace layer (full observability, never exposed)
│   │   ├── logger.py
│   │   ├── run_summary.py
│   │   └── schema.py
│   └── utils/
├── scripts/
│   ├── run_landscape.py     # Landscape characterization entry point
│   ├── run_grid.sh          # LLM experiment grid (144 runs)
│   └── run_random_baseline.sh  # Random search baseline (9 runs)
├── logs/
│   ├── landscape/           # Landscape results + init configs
│   ├── traces/
│   └── runs/
└── run_*_bench.py
```

## Quick Start

```bash
# 1. Landscape characterization (one-time per benchmark, no LLM needed)
python scripts/run_landscape.py --benchmark <benchmark> --n-configs 256

# 2. Single benchmark run
python run_<benchmark>_bench.py --num-steps 10 --show-task --show-metric

# Random search baseline (for comparison)
python run_<benchmark>_bench.py --num-steps 10 --optimizer random --seed 42

# 3. Full LLM experiment grid (144 runs: 3 init × 16 context × 3 seeds)
#    Requires landscape characterization (step 1) to have been run first.
./scripts/run_grid.sh <benchmark> --dry-run    # preview
./scripts/run_grid.sh <benchmark>              # run

# 4. Random search baseline (9 runs: 3 init × 3 seeds, zero API cost)
./scripts/run_random_baseline.sh --benchmarks <benchmark> --dry-run
./scripts/run_random_baseline.sh --benchmarks <benchmark>
```

Where `<benchmark>` is one of: `nomad`, `forest`, `housing`, `jigsaw`, `toy`.

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

### Optimizer Selection

| Flag | Default | Description |
|------|---------|-------------|
| `--optimizer` | llm | Optimization strategy: `llm` (LLM-based) or `random` (uniform sampling baseline) |

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

